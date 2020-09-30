from tensorflow import keras

from util.partition_strategy import GraphPartitionStrategy
from tf_src.agcn.layer import GraphConvolutionSequenceLayer


def create_model(config, graph, input_shape):
    strategy = GraphPartitionStrategy(config.strategy)
    adj = strategy.get_adjacency_matrix_array(graph)
    num_channels, num_frames, num_joints, num_bodies = input_shape

    model = keras.models.Sequential([
        # shape after transpose = (batch_size, num_bodies, num_joints, num_channels, num_frames)
        keras.layers.Permute((4, 3, 1, 2)),
        # Reshape to batch normalize over each batch and frame
        # Shape after reshape = (batch_size, num_bodies * num_joints * num_channels, num_frames)
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
