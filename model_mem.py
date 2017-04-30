import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split

samples = []
with open("./mydata/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.1)
correction = 1.0

def process_image(images, angles, path, angle, train, correction=0.0):
    filename = re.split(r'\\|/',path)[-1]
    name = "./mydata/IMG/" + filename
    image = cv2.imread(name)
    image_flipped = np.fliplr(image)
    angle_flipped = -angle
    images.append(image)
    angles.append(angle)
    if train:
        images.append(image_flipped)
        angles.append(angle_flipped)

def process_samples(samples, train):
    images = []
    angles = []
    for sample in samples:
        angle = float(sample[3])
        process_image(images, angles, sample[0], angle, train)
        if train:
            process_image(images, angles, sample[1], angle, train, correction)
            process_image(images, angles, sample[2], angle, train, -correction)
    X = np.array(images)
    y = np.array(angles)
    return X, y

# compile and train the model using the generator function
X, y = process_samples(train_samples, True)
valid_data = process_samples(validation_samples, False)

row, col, ch = 160, 320, 3
cropping_height = (70, 25)
cropping_width = (0, 0)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=(cropping_height, cropping_width)))
model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48, (5, 5), activation='relu'))
model.add(Convolution2D(64, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile('adam', 'mse')
history_object = model.fit(X, y, batch_size=128, epochs=7, verbose=1, validation_data=valid_data, shuffle=True)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

