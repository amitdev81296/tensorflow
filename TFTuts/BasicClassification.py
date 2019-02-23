from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper Libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images)
# print(train_labels)
# print(test_images)
# print(test_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(test_images.shape)
# print(len(test_labels))

# Preprocessing the data
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# Pixel values fall between 0 and 255. We scale these values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Displaying the first 25 images in the training set with their class name
# lt.figure(figsize = (10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_images[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build the model
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model with addition of few more settings
model.compile (
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the acuracy of the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print ('Test Accuracy : ' , test_acc)


# Making predictions. Here the model will predict the label for each image in the testing set
predictions = model.predict(test_images)

print("Predicted : ", np.argmax(predictions[0]), "\nActual : ", test_labels[0])
