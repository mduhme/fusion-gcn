import os
import time
from tqdm import tqdm
from sys import stdout
from itertools import chain

from tfmodels.agcn.config import get_config
from datasets.ntu_rgb_d.constants import skeleton_edges, data_shape
from util.graph import Graph

import numpy as np
import tensorflow as tf

# Enable memory growth after first time tensorflow import and before keras import
physical_devices = tf.config.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from tensorflow import keras
from tfmodels.agcn.model import create_model


def load_data(config):
    feature_description = {
        "features": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }

    @tf.function
    def parse_single_example(example):
        example_features = tf.io.parse_single_example(example, feature_description)
        features = tf.io.parse_tensor(example_features["features"], tf.float32)
        label = tf.one_hot(example_features["label"], config.num_classes)
        features = tf.reshape(features, data_shape)  # for some reason this line is required to prevent a crash
        return features, label

    training_set = tf.data.TFRecordDataset([os.path.join(config.in_path, "train_data.tfrecord")])
    test_set = tf.data.TFRecordDataset([os.path.join(config.in_path, "val_data.tfrecord")])

    training_samples = None
    test_samples = None
    if os.path.exists(os.path.join(config.in_path, "train_features.npy")):
        training_samples = len(np.load(os.path.join(config.in_path, "train_features.npy"), mmap_mode="r"))
    if os.path.exists(os.path.join(config.in_path, "val_features.npy")):
        test_samples = len(np.load(os.path.join(config.in_path, "val_features.npy"), mmap_mode="r"))

    # Map 'example' to features and label
    training_set = training_set.map(parse_single_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_set = test_set.map(parse_single_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not config.no_cache:
        training_set = training_set.cache()
        test_set = test_set.cache()

    if not config.disable_shuffle:
        shuffle_buffer_size = training_samples or 1000
        training_set = training_set.shuffle(shuffle_buffer_size)

    # Enable batching and prefetching
    training_set = training_set.batch(config.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.batch(config.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return training_set, test_set, training_samples, test_samples


@tf.function
def train_single_batch(model, x, y_true, optimizer, loss_function):
    """
    Default training step (for float32 precision).
    :param model: Model to train
    :param x: features
    :param y_true: ground truth labels
    :param optimizer: optimizer to be used
    :param loss_function: loss function to be used
    :return: A tuple (label predictions, loss)
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_function(y_true, y_pred)
        loss += tf.reduce_sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, loss


@tf.function
def test_single_batch(model, x, y_true, loss_function):
    y_pred = tf.nn.softmax(model(x, training=False))
    loss = loss_function(y_true, y_pred)
    loss += tf.reduce_sum(model.losses)
    return y_pred, loss


class TrainingMetric:
    def __init__(self, metric, name=None):
        self.metric = metric
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return self.metric.name

    @property
    def value(self):
        if isinstance(self.metric, keras.metrics.Metric):
            return self.metric.result()
        elif (isinstance(self.metric, tf.Variable) or isinstance(self.metric, tf.Tensor)) and tf.size(self.metric) == 1:
            return float(self.metric.numpy())
        raise ValueError("Invalid metric type")

    def kv(self):
        return self.name, self.value

    def reset(self):
        if isinstance(self.metric, keras.metrics.Metric):
            return self.metric.reset_states()


class ModelTraining:
    def __init__(self, config, model, train_set, validation_set, num_training_samples=None,
                 num_validation_samples=None, optimizer=None, loss_function=None):
        self.config = config
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.num_training_samples = num_training_samples
        self.num_validation_samples = num_validation_samples

        self.training_id = time.strftime("training_%Y_%m_%d-%H_%M_%S")
        self.log_path = os.path.join(self.config.log_path, self.training_id)
        self.check_point_path = os.path.join(self.config.checkpoint_path, self.training_id)
        if not os.path.exists(self.check_point_path):
            os.makedirs(self.check_point_path)

        # TODO store config in log dir for reproducibility

        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = keras.optimizers.SGD(learning_rate=self.config.base_lr, momentum=0.9, nesterov=True)
        self.loss_function = loss_function
        if loss_function is None:
            self.loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)

        steps = np.array(config.steps) - 2
        values = [config.base_lr ** i for i in range(1, len(config.steps) + 2)]
        self.lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(steps, values)
        self.best_checkpoints = []
        self.max_checkpoints_to_keep = 5

    def _create_trace(self, logger):
        """
        Create graph trace section in TensorBoard
        :param logger: file writer
        """
        features_batch, y_true = next(iter(self.train_set))
        print("Create training trace graph")
        tf.summary.trace_on(graph=True)
        train_single_batch(self.model, features_batch, y_true, self.optimizer, self.loss_function)
        with logger.as_default():
            tf.summary.trace_export("training_trace", step=0)
        tf.summary.trace_off()
        print("Create testing trace graph")
        tf.summary.trace_on(graph=True)
        test_single_batch(self.model, features_batch, y_true, self.loss_function)
        with logger.as_default():
            tf.summary.trace_export("testing_trace", step=0)
        tf.summary.trace_off()

    def _learning_rate_scheduling(self, epoch):
        """
        Learning rate scheduling.
        Currently, learning rate is divided by 10 at fixed epochs given by self.config.steps
        :param epoch: Current epoch
        """
        new_learning_rate = self.lr_scheduler(epoch)
        if self.optimizer.learning_rate != new_learning_rate:
            keras.backend.set_value(self.optimizer.learning_rate, new_learning_rate)

    def _maybe_create_checkpoint(self, accuracy: float, epoch: int):
        """
        Keeps track of the n best checkpoints and removes old checkpoints with worse results
        :param accuracy: metric for comparison (higher values will stay)
        :param epoch: epoch will be written to the file name
        :return:
        """

        def _create_path(check_point_path: str, acc: float, ep: int):
            return os.path.join(check_point_path, "checkpoint_{0:0>2}_{1:.4f}.h5".format(ep, acc))

        if len(self.best_checkpoints) >= self.max_checkpoints_to_keep and accuracy <= self.best_checkpoints[-1][0]:
            return

        self.best_checkpoints.append((accuracy, epoch))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        self.model.save(_create_path(self.check_point_path, accuracy, epoch))

        for cp in self.best_checkpoints[self.max_checkpoints_to_keep:]:
            p = _create_path(self.check_point_path, *cp)
            os.remove(p)

        self.best_checkpoints = self.best_checkpoints[:self.max_checkpoints_to_keep]

    def create_metrics(self):
        training_loss = keras.metrics.Mean(name="training_loss")
        validation_loss = keras.metrics.Mean(name="validation_loss")
        training_metrics = [
            keras.metrics.CategoricalAccuracy(name="training_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(name="training_top5_accuracy")
        ]
        validation_metrics = [
            keras.metrics.CategoricalAccuracy(name="validation_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(name="validation_top5_accuracy")
        ]

        metrics = [TrainingMetric(m) for m in
                   chain([training_loss], training_metrics, [validation_loss], validation_metrics)]
        metrics.append(TrainingMetric(self.optimizer.learning_rate, "lr"))

        return training_loss, validation_loss, training_metrics, validation_metrics, metrics

    def start(self):
        """
        Create metrics and start training.
        """

        training_loss, validation_loss, training_metrics, validation_metrics, metrics = self.create_metrics()
        metric_names = [m.name for m in metrics]
        logger = tf.summary.create_file_writer(self.log_path)
        self._create_trace(logger)

        num_batches = self.num_training_samples // self.config.batch_size if self.num_training_samples else None
        val_batches = self.num_validation_samples // self.config.batch_size if self.num_validation_samples else None

        # TODO maybe try different types of learning rate scheduling (1cycle, ...)
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}:")
            self._learning_rate_scheduling(epoch)

            print("Train batches...")
            bar = keras.utils.Progbar(num_batches, stateful_metrics=metric_names)
            step = 0

            for step, (features_batch, y_true) in enumerate(self.train_set):

                if epoch == 0 and self.config.profiling:
                    if step == self.config.profiling_range[0]:
                        tf.profiler.experimental.start(self.log_path)
                    elif step == self.config.profiling_range[1]:
                        tf.profiler.experimental.stop()

                # Forward and backward training pass
                y_pred, loss = train_single_batch(self.model, features_batch, y_true, self.optimizer,
                                                  self.loss_function)

                # Update training metrics
                training_loss.update_state(loss)
                for metric in training_metrics:
                    metric.update_state(y_true, y_pred)

                # Update progress bar
                bar.update(step, [metric.kv() for metric in metrics if metric.name.startswith("training")])

            for features_batch, y_true in tqdm(self.validation_set, desc="Validation", total=val_batches, leave=False,
                                               file=stdout):
                y_pred, loss = test_single_batch(model, features_batch, y_true, self.loss_function)

                # Update validation metrics
                validation_loss.update_state(loss)
                for metric in validation_metrics:
                    metric.update_state(y_true, y_pred)

            bar.update(step + 1, [metric.kv() for metric in metrics], finalize=True)

            with logger.as_default():
                for metric in metrics:
                    tf.summary.scalar(metric.name, metric.value, step=epoch)

            self._maybe_create_checkpoint(float(validation_metrics[0].result()), epoch)

            # Reset metric states
            for metric in metrics:
                metric.reset()


def train_model_old(config, train_set, test_set, num_training_samples, num_test_samples):
    graph = Graph(skeleton_edges, is_directed=True)
    shape = (config.batch_size, 3, 300, 25, 2)

    model = create_model(config, graph, shape)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=config.base_lr, momentum=0.9, nesterov=True),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy", "top_k_categorical_accuracy"])
    model.summary()

    lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(config.steps, [config.base_lr ** i for i in
                                                                                    range(1, len(config.steps) + 2)])

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_scheduler),
        keras.callbacks.TensorBoard(os.path.join(config.log_path, time.strftime("training_%Y_%m_%d-%H_%M_%S")),
                                    profile_batch="200,250"),
        keras.callbacks.ModelCheckpoint(os.path.join(config.checkpoint_path, "weights.{epoch:02d}.h5"),
                                        save_best_only=True)
    ]

    model.fit(train_set, epochs=config.epochs, validation_data=test_set, callbacks=callbacks)


if __name__ == "__main__":
    cf = get_config()
    setattr(cf, "kernel_regularizer", keras.regularizers.l2(cf.weight_decay))

    graph = Graph(skeleton_edges, is_directed=True)
    model = create_model(cf, graph, data_shape)

    training_procedure = ModelTraining(cf, model, *load_data(cf))
    training_procedure.start()
