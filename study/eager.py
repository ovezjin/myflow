from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
import tensorflow.contrib.eager as tfe 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager ececution: {}".format(tf.executing_eagerly()))

train_dataset_fp = "C:/Users/vg/Desktop/hccT.csv"
test_fp = "C:/Users/vg/Desktop/89852.csv"

def parse_csv(line):
	example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
	[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
	[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 
	[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
	[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
	[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 
	[0],]
	parsed_line = tf.decode_csv(line, example_defaults)
	features = tf.reshape(parsed_line[:-1], shape = (61,))
	label = tf.reshape(parsed_line[-1], shape = ())

	return features, label

def cdataset(fp):
	dataset = tf.data.TextLineDataset(fp)
	dataset = dataset.skip(1)
	dataset = dataset.map(parse_csv)
	dataset = dataset.shuffle(buffer_size = 1000)
	dataset = dataset.batch(4)
	return dataset

train_dataset = cdataset(train_dataset_fp)
test_dataset = cdataset(test_fp)

features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

model = tf.keras.Sequential([
	tf.keras.layers.Dense(16, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001), input_shape = (61,)),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(16, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(2, activation = 'sigmoid')
	])

def loss(model, x, y):
	y_ = model(x)
	return tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)

def grad(model, inputs, targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model, inputs, targets)
	return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

train_loss_results = []
train_accuracy_results = []
num_epochs = 801

for epoch in range(num_epochs):
	epoch_loss_avg = tfe.metrics.Mean()
	epoch_accuracy = tfe.metrics.Accuracy()

	for x, y in train_dataset:
		grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.variables), global_step = tf.train.get_or_create_global_step())
		epoch_loss_avg(loss(model, x, y))
		epoch_accuracy(tf.argmax(model(x), axis = 1, output_type = tf.int32), y)

	train_loss_results.append(epoch_loss_avg.result())
	train_accuracy_results.append(epoch_accuracy.result())
	if epoch % 50 == 0:
		print("Epoch {:04d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex = True, figsize = (12, 8))
fig.suptitle('Traing Metrics')
axes[0].set_ylabel("Loss", fontsize = 14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize = 14)
axes[1].set_xlabel("Epoch", fontsize = 14)
axes[1].plot(train_accuracy_results)
plt.show()

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
	prediction = tf.argmax(model(x), axis = 1, output_type = tf.int32)
	test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))