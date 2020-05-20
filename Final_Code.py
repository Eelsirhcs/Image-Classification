# CHRIS LEE, JIAHUI LI, MINGHAN SUN, WILLIAM SALAS

#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import keras
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# List of names for each CIFAR10 class
cifar10_class_names = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}


# In[ ]:


# Data Exploratory
n = 6
with plt.style.context('default'):
    plt.figure(num = 1, figsize = [10, 10])
    for i in range(2 * n):
        # Grab an image from the data set
        sample_image = x_train[i]
        # Grab the image's expected class id
        image_class_number = y_train[i][0]
        # Look up the string name of that label from the list of labels we have at the top
        image_class_name = cifar10_labels[image_class_number]

        # Draw the image as a plot
        plt.subplot(n, n, i+1)
        plt.imshow(sample_image)
        # Label the image
        plt.title(image_class_name)


# In[ ]:


#We compile and we define how we'll be training it and how we'll be measuring it accuracy.
#############################################
# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Pre-Processing
# Before we use the data to train a neural network, we need to normalize it.
# Neural networks work best when the input data are floating values between 0 and 1.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Since we are creating a neural network with 10 oututs, we create a separate expected value for each of the outputs
## Convert class vectors to binary class matrices
# Our labels are single values from 0 to 9.
# Instead, we want each label to be an array with on element set to 1 and and the rest set to 0.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (32, 32, 3), activation = "relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding = 'same', activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = "relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding = 'same', activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = "relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding = 'same', activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Compile the model (several parameters)
#loss function tells keras how to check how right or how wrong the guesses from out neural network are.
#Since we have 10 different possible categories for our objects with out data set, we'll use categorical crossentropy.
#Binary if we're only checking if an image belongs to one category.
#Choosing an optimization algorithm, adam = Adaptive Moment Estimation, often used for image data.
#The metrics we want the model to report during the training process, we type it into an array because we can choose to report more than one.
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


# Print a summary of the model
model.summary()


# In[ ]:


# Train the model
model.fit(x_train, y_train, batch_size = 64, epochs = 60, validation_data = (x_test, y_test), shuffle = True)

# Save neural network structure
model_structure = model.to_json()
f = Path("60Epochs.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("60Epochs.h5")


# In[ ]:


#Classification Report
y_true = np.dot(y_test, list(range(10)))
y_pred = model.predict_classes(x_test)
report = metrics.classification_report(y_true, y_pred)
print(report)
print(metrics.confusion_matrix(y_true, y_pred))
#print(model.summary())


# In[ ]:


#Loss and Accuracy
# plot loss during training
plt.style.use('ggplot')
base = 3.5
fig = plt.figure(num = 1, figsize = [2 * base * (1 + np.sqrt(5))/2, base])
plt.style.use('ggplot')
c1 = '0.5'
c2 = 'C0'
plt.subplot(121)
plt.title('Loss')
plt.plot(model.history.history['loss'], label='train', color = c1, linestyle = '--')
plt.plot(model.history.history['val_loss'], label='test', color = c2)
plt.legend(shadow = True, facecolor = '1')
# plot accuracy during training
plt.subplot(122)
plt.title('Accuracy')
plt.plot(model.history.history['accuracy'], label='train', color = c1, linestyle = '--')
plt.plot(model.history.history['val_accuracy'], label='test', color = c2)
plt.legend(shadow = True, facecolor = '1')
plt.show()

