from tensorflow import keras

from partition_strategy import GraphPartitionStrategy
from layer import GraphConvolutionSequenceLayer


class GraphSpatialBatchNormalization(keras.layers.Layer):
    """
    BatchNormalization for each individual frame.
    Expects input of shape (batch_size, num_bodies, num_joints, num_channels, num_frames)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inner_layers = []

    def build(self, input_shape):
        batch_size, num_bodies, num_joints, num_channels, num_frames = input_shape
        self.inner_layers = [
            # Reshape to batch normalize over each batch and frame
            # Shape after reshape = (batch_size, num_bodies * num_joints * num_channels, num_frames)
            keras.layers.Reshape((-1, num_frames)),
            keras.layers.BatchNormalization(axis=1, epsilon=1e-5),
            # Reshape back
            keras.layers.Reshape((num_bodies, num_joints, num_channels, num_frames)),
        ]

    def call(self, inputs, training=None, **kwargs):
        for layer in self.inner_layers:
            inputs = layer(inputs, training=training, **kwargs)
        return inputs


def create_model(config, graph, input_shape):
    strategy = GraphPartitionStrategy(config.strategy)
    adj = strategy.get_adjacency_matrix_array(graph)
    num_channels, num_frames, num_joints, num_bodies = input_shape

    model = keras.models.Sequential([
        # shape after transpose = (batch_size, num_bodies, num_joints, num_channels, num_frames)
        keras.layers.Permute((4, 3, 1, 2)),
        keras.layers.Reshape((-1, num_frames)),
        keras.layers.BatchNormalization(axis=1, epsilon=1e-5),
        # Reshape back
        keras.layers.Reshape((num_bodies, num_joints, num_channels, num_frames)),
        # shape after transpose = (batch_size, num_bodies, num_channels, num_frames, num_joints)
        keras.layers.Permute((1, 3, 4, 2)),

        GraphConvolutionSequenceLayer(config, adj),

        keras.layers.Dense(config.num_classes, kernel_regularizer=config.kernel_regularizer)
    ], "AGCN")

    return model
