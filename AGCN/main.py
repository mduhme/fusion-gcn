import os
import time
from tqdm import tqdm
from sys import stdout

from config import get_config
from datasets.ntu_rgb_d.constants import skeleton_edges, data_shape
from util.graph import Graph

import numpy as np
import tensorflow as tf

# Enable memory growth after first time tensorflow import and before keras import
physical_devices = tf.config.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from tensorflow import keras
from model import create_model


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

    if not config.no_shuffle:
        training_set = training_set.shuffle(1000)

    # Enable batching and prefetching
    training_set = training_set.batch(config.batch_size, drop_remainder=True).prefetch(1)
    test_set = test_set.batch(config.batch_size).prefetch(1)

    return training_set, test_set, training_samples, test_samples


def create_learning_rate_scheduler(config):
    values = [config.base_lr ** i for i in range(1, len(config.steps) + 2)]
    return keras.optimizers.schedules.PiecewiseConstantDecay(config.steps, values)


@tf.function
def train_single_batch(model, x, y_true, optimizer, loss_function):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_function(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, loss


@tf.function
def test_single_batch(model, x, y_true, loss_function):
    y_pred = tf.nn.softmax(model(x, training=False))
    return y_pred, loss_function(y_true, y_pred)


class ModelTraining:
    def __init__(self, config, model, train_set, validation_set, num_training_samples=None,
                 num_validation_samples=None):
        self.config = config
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.num_training_samples = num_training_samples
        self.num_validation_samples = num_validation_samples

        self.training_id = time.strftime("training_%Y_%m_%d-%H_%M_%S")
        self.log_path = os.path.join(self.config.log_path, self.training_id)
        self.check_point_path = os.path.join(self.config.checkpoint_path, self.training_id)

    def _create_trace(self, logger, optimizer, loss_function):
        for features_batch, y_true in self.train_set:
            tf.summary.trace_on(graph=True)
            train_single_batch(self.model, features_batch, y_true, optimizer, loss_function)
            with logger.as_default():
                tf.summary.trace_export("training_trace", step=0)
            tf.summary.trace_off()
            tf.summary.trace_on(graph=True)
            test_single_batch(self.model, features_batch, y_true, loss_function)
            with logger.as_default():
                tf.summary.trace_export("testing_trace", step=0)
            tf.summary.trace_off()
            break

    def start(self, optimizer=None, loss_function=None):
        if optimizer is None:
            optimizer = keras.optimizers.SGD(learning_rate=self.config.base_lr, momentum=0.9, nesterov=True)
        if loss_function is None:
            loss_function = keras.losses.CategoricalCrossentropy(from_logits=True)

        training_loss = keras.metrics.Mean("training_loss")
        validation_loss = keras.metrics.Mean("validation_loss")
        training_metrics = [
            keras.metrics.CategoricalAccuracy(name="training_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(name="training_top5_accuracy")
        ]
        validation_metrics = [
            keras.metrics.CategoricalAccuracy(name="validation_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(name="validation_top5_accuracy")
        ]

        metrics = [training_loss, *training_metrics, validation_loss, *validation_metrics]
        metric_names = [metric.name for metric in metrics]

        logger = tf.summary.create_file_writer(self.log_path)
        self._create_trace(logger, optimizer, loss_function)

        num_batches = self.num_training_samples // self.config.batch_size if self.num_training_samples else None
        val_batches = self.num_validation_samples // self.config.batch_size if self.num_validation_samples else None

        # TODO add learning rate scheduling
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}:")
            bar = keras.utils.Progbar(num_batches, stateful_metrics=metric_names)

            print("Train batches...")
            step = 0

            for step, (features_batch, y_true) in enumerate(self.train_set):

                if epoch == 0 and self.config.profiling:
                    if step == self.config.profiling_range[0]:
                        tf.profiler.experimental.start(self.log_path)
                    elif step == self.config.profiling_range[1]:
                        tf.profiler.experimental.stop()

                # Forward and backward training pass
                y_pred, loss = train_single_batch(self.model, features_batch, y_true, optimizer, loss_function)

                # Update training metrics
                training_loss.update_state(loss)
                for metric in training_metrics:
                    metric.update_state(y_true, y_pred)

                # Update progress bar
                bar.update(step, [(metric.name, metric.result()) for metric in [training_loss] + training_metrics])

            for features_batch, y_true in tqdm(self.validation_set, total=val_batches, leave=False, file=stdout):
                y_pred, loss = test_single_batch(model, features_batch, y_true, loss_function)

                # Update validation metrics
                validation_loss.update_state(loss)
                for metric in validation_metrics:
                    metric.update_state(y_true, y_pred)

            bar.update(step + 1, [(metric.name, metric.result()) for metric in metrics], finalize=True)

            with logger.as_default():
                for metric in metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)

            # Reset metric states
            for metric in metrics:
                metric.reset_states()


def train_model_old(config, train_set, test_set, num_training_samples, num_test_samples):
    graph = Graph(skeleton_edges, is_directed=True)
    shape = (config.batch_size, 3, 300, 25, 2)

    model = create_model(config, graph, shape)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=config.base_lr, momentum=0.9, nesterov=True),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy", "top_k_categorical_accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.LearningRateScheduler(create_learning_rate_scheduler(config)),
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
    shape = (3, 300, 25, 2)
    model = create_model(cf, graph, shape)

    training_procedure = ModelTraining(cf, model, *load_data(cf))
    training_procedure.start()
