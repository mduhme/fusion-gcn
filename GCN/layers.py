import tensorflow as tf
from tensorflow import keras


class GraphConvolution(keras.layers.Layer):
    def __init__(self, units, support, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        assert type(units) is int and units > 0
        assert type(support) is list and len(support) > 0
        self.units = units
        self.support = support
        self.activation = keras.activations.get(kwargs.get("activation"))
        self.use_bias = kwargs.get("use_bias", True)
        self.dropout_rate = dropout_rate
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # Create weights for each support (chebyshev polynomial)
        self.kernel = [
            self.add_weight(name=f"kernel_{i}", shape=(input_shape[-1], self.units), initializer="glorot_uniform") for i
            in range(len(self.support))]

        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.dropout_rate is not None and self.dropout_rate > 0.:
            # TODO sparse version
            x = keras.layers.Dropout(self.dropout_rate)(x)

        summands = []
        for kernel, support in zip(self.kernel, self.support):
            # TODO sparse version
            # print("Input:", x.shape, x.dtype)
            # print("Kernel:", tf.expand_dims(kernel, 0).shape, tf.expand_dims(kernel, 0).dtype)
            res = tf.linalg.matmul(x, kernel)
            # print("Support:", tf.expand_dims(support, 0).shape, tf.expand_dims(support, 0).dtype)
            # print("Res:", res.shape, res.dtype)
            res = tf.linalg.matmul(support, res)
            summands.append(res)

        if len(summands) > 1:
            result = keras.layers.add(summands)
        else:
            result = summands[0]

        if self.use_bias:
            result = tf.nn.bias_add(result, self.bias)

        return self.activation(result)
