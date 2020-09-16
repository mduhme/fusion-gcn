import tensorflow as tf
from tensorflow import keras

INITIALIZER = keras.initializers.VarianceScaling(scale=2., mode="fan_out")


class SpatialGraphConvolution(keras.layers.Layer):
    """
    Layer implemented as in Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition
    Figure 2

    Pseudocode:
    f_out = sum(W[k] * f_in * (A[k] + B[k] + C[k]) for k in range(num_subsets))
    where
    - A[k] is the constant adjacency matrix of the kth subset
    - B[k] is a learnable weight matrix
    - C[k] = softmax(transpose(f_in) * transpose(W_theta[k]) * W_phi[k] * f_in), a data-dependent weight matrix that
    represents the similarity between vertex v_i and vertex v_j
    - W[k], W_theta[k], W_phi[k] are weights

    """

    def __init__(self, config, adjacency_matrix, num_filters, activation="relu", down=False, **kwargs):
        super().__init__(**kwargs)
        # Shape of adjacency matrix is (num_subsets, V, V) with V being the number of nodes
        self.activation = keras.activations.get(activation)
        self.num_subsets = int(tf.shape(adjacency_matrix)[0].numpy())

        self.a = adjacency_matrix  # Adjacency matrix A_k
        self.b = None  # Initialize in build to create unique name

        branch_conv_init = keras.initializers.VarianceScaling(scale=2. / self.num_subsets, mode="fan_out")
        num_inter_channels = num_filters // 4  # Why 4 is not explained in paper/code.

        # Create weights W[k], W_theta[k], W_phi[k] for each subset k
        # W_theta[k] and W_phi[k] are used to transform the input to an embedding space
        self.w = []
        self.w_theta = []
        self.w_phi = []
        for _ in range(self.num_subsets):
            self.w_theta.append(keras.layers.Conv2D(num_inter_channels, 1, padding="same", data_format="channels_first",
                                                    kernel_initializer=INITIALIZER,
                                                    kernel_regularizer=config.kernel_regularizer))
            self.w_phi.append(keras.layers.Conv2D(num_inter_channels, 1, padding="same", data_format="channels_first",
                                                  kernel_initializer=INITIALIZER,
                                                  kernel_regularizer=config.kernel_regularizer))
            self.w.append(keras.layers.Conv2D(num_filters, 1, padding="same", data_format="channels_first",
                                              kernel_initializer=branch_conv_init,
                                              kernel_regularizer=config.kernel_regularizer))
        # TODO only w uses 'branch_conv_init'

        self.bn = keras.layers.BatchNormalization(axis=1, epsilon=1e-5)
        self.down = down

        if down:
            self.down_conv = keras.layers.Conv2D(num_filters, 1, padding="same", data_format="channels_first",
                                                 kernel_initializer=INITIALIZER,
                                                 kernel_regularizer=config.kernel_regularizer)
            self.down_bn = keras.layers.BatchNormalization(axis=1)

    def build(self, input_shape):
        self.b = self.add_weight(name="adjacency_matrix_b", shape=tf.shape(self.a),
                                 initializer=keras.initializers.Constant(1e-6))

    def _sum_iteration(self, x, k):
        batch_size, num_channels, num_frames, num_joints = tf.unstack(tf.shape(x))
        c1 = tf.reshape(tf.transpose(self.w_theta[k](x), perm=(0, 3, 1, 2)),
                        (batch_size, num_joints, -1))  # TODO switch reshape and transpose
        c2 = tf.reshape(self.w_phi[k](x), (batch_size, -1, num_joints))
        # divide first matrix now to prevent overflow
        c1 /= tf.cast(tf.shape(c1)[-1], c1.dtype)
        c = tf.matmul(c1, c2)
        # Original line (results in overflow with mixed precision):
        # c = c / tf.cast(tf.shape(c1)[-1], c.dtype)
        c = tf.nn.softmax(c, axis=-2)
        x_out = tf.matmul(tf.reshape(x, (batch_size, -1, num_joints)), self.a[k] + self.b[k] + c)
        x_out = tf.reshape(x_out, tf.shape(x))
        return self.w[k](x_out)

    def call(self, inputs, training=None, **kwargs):
        x = tf.reduce_sum([self._sum_iteration(inputs, k) for k in range(self.num_subsets)], axis=0)
        x = self.bn(x, training=training)
        if self.down:
            res = self.down_conv(inputs, training=training)
            res = self.down_bn(res, training=training)
            x += res
        return self.activation(x)


class TemporalGraphConvolution(keras.layers.Layer):
    def __init__(self, config, num_filters, kernel_size, stride, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)  # TODO use this? It is unused in original paper
        # Input shape is (batch_size*num_bodies, num_channels, num_frames, num_joints)
        self.layers = [
            # batch normalize over channels
            keras.layers.Conv2D(num_filters, [kernel_size, 1], [stride, 1], padding="same",
                                kernel_initializer=INITIALIZER, data_format="channels_first",
                                kernel_regularizer=config.kernel_regularizer),
            keras.layers.BatchNormalization(axis=1, epsilon=1e-5)
        ]

    def call(self, inputs, training=None, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training, **kwargs)
        return inputs


class GraphConvolution(keras.layers.Layer):
    def __init__(self, config, num_filters, adjacency_matrix, temporal_kernel_size=9, activation="relu",
                 temporal_stride=1, residual=True, down=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.spatial_layer = SpatialGraphConvolution(config, adjacency_matrix, num_filters, down=down, **kwargs)
        self.temporal_layer = TemporalGraphConvolution(config, num_filters, temporal_kernel_size, temporal_stride,
                                                       **kwargs)
        self.residual_layers = []

        # TODO parameter 'downsampling' unnecessary
        if not residual:
            self.residual = lambda x, training=None: 0
        elif down and temporal_stride == 1:
            self.residual = lambda x, training=None: x
        else:
            self.residual = TemporalGraphConvolution(config, num_filters, 1, temporal_stride, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.spatial_layer(x, training=training, **kwargs)
        x = self.temporal_layer(x, training=training, **kwargs)
        x += self.residual(inputs, training=training, **kwargs)
        return self.activation(x)


class GraphConvolutionSequenceLayer(keras.layers.Layer):
    """
    A sequence of graph convolution layers with global average pooling at the end.
    """

    def __init__(self, config, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        # Add Graph Convolution layers and finalize with global average pooling
        self.adjacency_matrix = adjacency_matrix
        self.inner_layers = [
            GraphConvolution(config, 64, adjacency_matrix, residual=False, **kwargs),
            GraphConvolution(config, 64, adjacency_matrix, **kwargs),
            GraphConvolution(config, 64, adjacency_matrix, **kwargs),
            GraphConvolution(config, 64, adjacency_matrix, **kwargs),
            GraphConvolution(config, 128, adjacency_matrix, temporal_stride=2, down=True, **kwargs),
            GraphConvolution(config, 128, adjacency_matrix, **kwargs),
            GraphConvolution(config, 128, adjacency_matrix, **kwargs),
            GraphConvolution(config, 256, adjacency_matrix, temporal_stride=2, down=True, **kwargs),
            GraphConvolution(config, 256, adjacency_matrix, **kwargs),
            GraphConvolution(config, 256, adjacency_matrix, **kwargs),
        ]

    def compute_output_shape(self, input_shape):
        return input_shape[0], 256, 1, 1

    def call(self, inputs, training=None, **kwargs):
        batch_size, num_bodies = tf.unstack(tf.shape(inputs))[:2]
        # Merge batch_size and num_bodies: Output shape = (batch_size*num_bodies, num_channels, num_frames, num_joints)
        # Required for using the Conv2D layers that expect
        # (batch_size, num_channels, W, H) with data_format="channels_first"
        # TODO Don't like modifying the batch_size, maybe it should be merged with another property (num_joints?)
        x = tf.reshape(inputs, tf.concat([tf.reduce_prod(tf.shape(inputs)[:2], keepdims=True), tf.shape(inputs)[2:]],
                                         axis=0))

        for layer in self.inner_layers:
            x = layer(x, training=training, **kwargs)

        # Reshape for each batch and body:
        x = tf.reshape(x, (batch_size, num_bodies, tf.shape(x)[1], -1))
        # Mean over each body:
        x = tf.reduce_mean(tf.reduce_mean(x, axis=3), axis=1)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "adjacency_matrix": self.adjacency_matrix}
