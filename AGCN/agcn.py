from util.graph import Graph
import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2., seed=123,
                                                    mode='fan_out',
                                                    distribution='truncated_normal')

'''
Temporal Convolutional layer
Args:
  filters     : number of filters/output units
  kernel_size : kernel size for conv layer
  stride      : stride for conv layer
Returns:
  A Keras model instance for the block.
'''
class TCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=9, stride=1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters,
                                           kernel_size=[kernel_size, 1],
                                           strides=[stride, 1],
                                           padding='same',
                                           kernel_initializer=INITIALIZER,
                                           data_format='channels_first',
                                           kernel_regularizer=REGULARIZER)
        self.bn   = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, x, training):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class GCN(tf.keras.Model):
    def __init__(self, filters, adjacency_matrix, coff_embedding=4, num_subset=3,
                 down=False):
        super().__init__()
        self.num_subset = num_subset
        self.down = down
        inter_channels = filters//coff_embedding
        tf.random.set_seed(123)
        branch_conv_init = tf.keras.initializers.VarianceScaling(scale=2./self.num_subset,
                                                                 mode='fan_out', seed=123,
                                                                 distribution='truncated_normal')

        self.conv_a = []
        self.conv_b = []
        self.conv_d = []
        for _ in range(self.num_subset):
            self.conv_a.append(tf.keras.layers.Conv2D(inter_channels,
                                                      kernel_size=1,
                                                      padding='same',
                                                      kernel_initializer=branch_conv_init,
                                                      data_format='channels_first',
                                                      kernel_regularizer=REGULARIZER))
            self.conv_b.append(tf.keras.layers.Conv2D(inter_channels,
                                                      kernel_size=1,
                                                      padding='same',
                                                      kernel_initializer=branch_conv_init,
                                                      data_format='channels_first',
                                                      kernel_regularizer=REGULARIZER))
            self.conv_d.append(tf.keras.layers.Conv2D(filters,
                                                      kernel_size=1,
                                                      padding='same',
                                                      kernel_initializer=branch_conv_init,
                                                      data_format='channels_first',
                                                      kernel_regularizer=REGULARIZER))

        self.B = tf.Variable(initial_value=tf.ones_like(adjacency_matrix, dtype=tf.float32)*1e-6,
                             trainable=True,
                             name='parametric_adjacency_matrix')

        self.A = tf.Variable(initial_value=adjacency_matrix,
                             trainable=False, dtype=tf.float32,
                             name='adjacency_matrix')

        self.bn = tf.keras.layers.BatchNormalization(axis=1)

        if self.down:
            self.conv_down = tf.keras.layers.Conv2D(filters,
                                                    kernel_size=1,
                                                    padding='same',
                                                    kernel_initializer=INITIALIZER,
                                                    data_format='channels_first',
                                                    kernel_regularizer=REGULARIZER)
            self.bn_down = tf.keras.layers.BatchNormalization(axis=1)


    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x_agg = []
        for i in range(self.num_subset):
            C1 = self.conv_a[i](x)
            C1 = tf.transpose(C1, perm=(0, 3, 1, 2))
            C1 = tf.reshape(C1, (N, V, -1))

            C2 = self.conv_b[i](x)
            C2 = tf.reshape(C2, (N, -1, V))

            C_comb = tf.matmul(C1, C2) / tf.cast(tf.shape(C1)[-1], dtype=tf.float32)
            C_comb = tf.nn.softmax(C_comb, axis=-2) # N V V

            x_k = tf.reshape(x, (N, -1, V))
            x_k = tf.matmul(x_k, self.A[i] + self.B[i] + C_comb)
            x_k = tf.reshape(x_k, (N, C, T, V))
            x_k = self.conv_d[i](x_k)
            x_agg.append(x_k)

        x_agg = tf.reduce_sum(x_agg, axis=0)
        x_agg = self.bn(x_agg, training=training)
        if self.down:
            x_agg += self.bn_down(self.conv_down(x), training=training)
        x_agg = tf.nn.relu(x_agg)
        return x_agg


'''
Graph Temporal Convolutional Layer
Args:
  filters          : number of filters/output units
  adjacency_matrix : Adjacency matrix
  stride           : stride for convolutional layers
  residual         : Enables skip connections
  down             : Enables conv on skip connection
Returns:
  A Keras model instance for the block.
'''
class GraphTemporalConv(tf.keras.Model):
    def __init__(self, filters, adjacency_matrix, stride=1, residual=True,
                 down=False):
        super().__init__()
        self.gcn = GCN(filters, adjacency_matrix, down=down)
        self.tcn = TCN(filters, stride=stride)

        if not residual:
            self.residual = lambda x, training=False: 0
        elif down and (stride == 1):
            self.residual = lambda x, training=False: x
        else:
            self.residual = TCN(filters, kernel_size=1, stride=stride)

    def call(self, x, training):
        skip = x
        x = self.gcn(x, training=training)
        x = self.tcn(x, training=training)
        x += self.residual(skip, training=training)
        x = tf.keras.activations.relu(x)
        return x


'''
Graph Temporal Convolutional Layer
Args:
  num_class : number of classes for final output size
Returns:
  A Keras model instance for the block.
'''
class AGCN(tf.keras.Model):
    def __init__(self, num_classes=60):
        super().__init__()

        from partition_strategy import GraphPartitionStrategy
        from datasets.ntu_rgb_d.constants import skeleton_edges
        graph = Graph(skeleton_edges, is_directed=True)
        strategy = GraphPartitionStrategy()
        A = strategy.get_adjacency_matrix_array(graph)

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.GTC_layers = []
        self.GTC_layers.append(GraphTemporalConv(64,  A, down=False, residual=False))
        self.GTC_layers.append(GraphTemporalConv(64,  A))
        self.GTC_layers.append(GraphTemporalConv(64,  A))
        self.GTC_layers.append(GraphTemporalConv(64,  A))
        self.GTC_layers.append(GraphTemporalConv(128, A, down=True, stride=2))
        self.GTC_layers.append(GraphTemporalConv(128, A))
        self.GTC_layers.append(GraphTemporalConv(128, A))
        self.GTC_layers.append(GraphTemporalConv(256, A, down=True, stride=2))
        self.GTC_layers.append(GraphTemporalConv(256, A))
        self.GTC_layers.append(GraphTemporalConv(256, A))

        self.fc = tf.keras.layers.Dense(num_classes,
                                        kernel_initializer=INITIALIZER,
                                        kernel_regularizer=REGULARIZER)

    def call(self, x, training):
        BatchSize = tf.shape(x)[0]
        C         = tf.shape(x)[1]
        T         = tf.shape(x)[2]
        V         = tf.shape(x)[3]
        M         = tf.shape(x)[4]

        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [BatchSize, -1, T])
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [BatchSize, M, V, C, T])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [BatchSize * M, C, T, V])

        for layer in self.GTC_layers:
            x = layer(x, training=training)

        # N*M,C,T,V
        c_new = tf.shape(x)[1]
        x = tf.reshape(x, [BatchSize, M, c_new, -1])
        x = tf.reduce_mean(tf.reduce_mean(x, axis=3), axis=1)
        x = self.fc(x)

        return x
