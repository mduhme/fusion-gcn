import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from tensorflow import keras
from tfmodels.gcn.utils import *
from tfmodels.gcn.layers import GraphConvolution
import argparse

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

print("Tensorflow version", tf.__version__)

args = argparse.ArgumentParser()
args.add_argument("--model", default="gcn")
args.add_argument("--learning_rate", default=0.01)
args.add_argument("--epochs", default=200)
args.add_argument("--hidden1", default=16)
args.add_argument("--dropout", default=0.5)
args.add_argument("--weight_decay", default=5e-4)
args.add_argument("--early_stopping", default=10)
args.add_argument("--max_degree", default=3)
args = args.parse_args()
print(args)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()
print("adj:", adj.shape)
print("features:", features.shape)
print("y:", y_train.shape, y_val.shape, y_test.shape)
print("mask:", train_mask.shape, val_mask.shape, test_mask.shape)

features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
print("features coordinates::", features[0].shape)
print("features data::", features[1].shape)
print("features shape::", features[2])

if args.model == "gcn":
    # D^-0.5 A D^-0.5
    support = [preprocess_adj(adj)]
elif args.model == "gcn_cheby":
    support = chebyshev_polynomials(adj, args.max_degree)
else:
    raise ValueError("Invalid argument for model: " + str(args.model))

train_label = tf.convert_to_tensor(y_train)
train_mask = tf.convert_to_tensor(train_mask)
val_label = tf.convert_to_tensor(y_val)
val_mask = tf.convert_to_tensor(val_mask)
test_label = tf.convert_to_tensor(y_test)
test_mask = tf.convert_to_tensor(test_mask)
features = tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(*features)))
support = [tf.sparse.to_dense(tf.sparse.reorder(tf.cast(tf.SparseTensor(*support[0]), dtype=tf.float32)))]

X_train = tf.boolean_mask(features, train_mask, axis=0)
X_val = tf.boolean_mask(features, val_mask, axis=0)

print("X_train.shape", X_train.shape)
print("train_mask.shape", train_mask.shape)
print("X_val.shape", X_val.shape)
print("train_label", train_label.shape)
print("val_label", val_label.shape)

input_dim = features.shape
hidden_dim = args.hidden1
output_dim = y_train.shape[1]
dropout_rate = args.dropout

print("Input dim:", input_dim)

features = tf.expand_dims(features, 0)
model = keras.models.Sequential([
    GraphConvolution(hidden_dim, support, dropout_rate=dropout_rate, input_shape=input_dim),
    GraphConvolution(output_dim, support, dropout_rate=dropout_rate)
])
optimizer = keras.optimizers.Adam(lr=args.learning_rate)


def gcn_loss(model, y_pred, y, mask, weight_decay):
    l2_loss = sum(weight_decay * tf.nn.l2_loss(v) for v in model.layers[0].trainable_variables)
    y = tf.boolean_mask(y, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    cross_entropy = keras.losses.categorical_crossentropy(y, y_pred, True)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return l2_loss + cross_entropy


def gcn_accuracy(y_pred, y, mask):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


for epoch in range(1, args.epochs + 1):
    with tf.GradientTape() as tape:
        y_pred = model(features)
        y_pred = tf.squeeze(y_pred, axis=0)
        main_loss = tf.add_n(model.losses + [gcn_loss(model, y_pred, train_label, train_mask, args.weight_decay)])
    gradients = tape.gradient(main_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy = gcn_accuracy(y_pred, train_label, train_mask)
    val_accuracy = gcn_accuracy(y_pred, val_label, val_mask)

    print("Epoch: ", epoch, "/", args.epochs, "Training loss:", main_loss.numpy(), "|", "Training Accuracy:",
          train_accuracy.numpy(), "|", "Val Accuracy:",
          val_accuracy.numpy())

test_accuracy = gcn_accuracy(y_pred, test_label, test_mask)
print("Test accuracy:", test_accuracy.numpy())


def test():
    x = np.arange(9).reshape((3, 3))
    x = np.array([x, x, x])
    print(x)
    r = tf.keras.layers.Dot(axes=(2, 1))([x, x])
    print(r)
    r = tf.linalg.matmul(x, x)
    tf.sparse.sparse_dense_matmul()
    print(r)


if __name__ == "__main__":
    # test()
    pass
