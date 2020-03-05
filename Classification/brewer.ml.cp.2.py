'''
A tensorflow neural network model for the MNIST dataset classification problem.

code based on: https://www.tensorflow.org/tutorials/keras/classification

'''
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import random as r
import matplotlib.pyplot as plt

SHOW_PLOTS = False
maxaccuracy = 0

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


############################################################
class_names = []
with open("ecoli.data", "r") as f:
  lines = f.readlines()
  newLines = []
  y = []
  for line in lines:
      for i in range(3):
        line = line.replace("  ", " ")
      cn = line.split(" ")
      if not cn[-1] in class_names:
        class_names.append(cn[-1])
      cn = [x.strip(' ') for x in cn]
      newLines.append([float(i) for i in cn[1:-2]])
      y.append(cn[-1])
      
Y = np.array(y)
for value in range(len(Y)):
  Y[value] = class_names.index(Y[value])
print(Y)

data = np.array(newLines)


#while True:
# LOAD IN THE DATA
"""mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
m = train_images.shape[0]
#print(train_images.shape)
#print(len(train_labels))
#print(test_images.shape)

#plot the first image
if SHOW_PLOTS:
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()"""

#plot the first 25 images with 'y' values
'''if SHOW_PLOTS:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()'''
      

# BUILD THE MODEL
activations = ["selu", "relu", "sigmoid", "tanh", "linear"]
model = keras.Sequential([
    keras.layers.Dense(7, activation=tf.nn.relu),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='sigmoid'),
    keras.layers.Dense(10, activation='tanh'),
    keras.layers.Dense(10, activation='linear'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(9, activation='softmax')   
])
"""keras.layers.Dense(500 + r.randint(0, 500), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(500 + r.randint(0, 500), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(128 + r.randint(0, 150), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(128 + r.randint(0, 150), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(100 + r.randint(0, 100), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(100 + r.randint(0, 100), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(10 + r.randint(0, 5), activation=activations[r.randint(0, len(activations) - 1)]),
keras.layers.Dense(10 + r.randint(0, 5), activation=activations[r.randint(0, len(activations) - 1)]),"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TRAIN THE MODEL
model.fit(data, Y, epochs=300, verbose=1)

# DETERMINE MODEL QUALITY RESULTS
test_loss, test_acc = model.evaluate(data,  Y, verbose=2)
print('\nTest accuracy:', test_acc)

# MAKE PREDICTIONS ON THE TEST SET 
#predictions = model.predict(test_images)
#print(predictions[0])

#print("First 100 predictions:")
#for i in range(100):
#    print("Prediction:",np.argmax(predictions[i]),"    Actual:",test_labels[i],"     Diff:",abs(np.argmax(predictions[i]) - int(test_labels[i])))

#display a random subset of 15 predictions.
"""m = test_images.shape[0]
if SHOW_PLOTS:
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    startTest = r.randint(0,m-(num_images*2))
    j = 0
    for i in range(startTest,startTest+num_images):
        plt.subplot(num_rows, 2*num_cols, 2*j+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*j+2)
        plot_value_array(i, predictions[i], test_labels)
        j += 1
    plt.tight_layout()
    plt.show()
if (test_acc > maxaccuracy):
  maxaccuracy = test_acc
  with open("best.txt", "w") as f:
    f.write(str(maxaccuracy))"""