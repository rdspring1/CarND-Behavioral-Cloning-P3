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

# Read lines from CSV file
samples = []
with open("../data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split into train and test datasets
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

# Training / Validation parameters
batch_size = 32
spe = int(6*len(train_samples) / batch_size)
vs = int(len(validation_samples) / batch_size)

# Left/Right steering angle correction
correction = 1.0

# Read Image and Steering Angle
def process_image(images, angles, path, angle, train, correction=0.0):
    filename = re.split(r'\\|/',path)[-1]
    name = "../data/IMG/" + filename
    image = cv2.imread(name)
    images.append(image)
    angles.append(angle)
    if train:
        image_flipped = np.fliplr(image)
        angle_flipped = -angle
        images.append(image_flipped)
        angles.append(angle_flipped)

def generator(samples, train, batch_size):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                process_image(images, angles, batch_sample[0], angle, train)
                if train:
					# Add left and right corrected images during traing
                    process_image(images, angles, batch_sample[1], angle, train, correction)
                    process_image(images, angles, batch_sample[2], angle, train, -correction)

                if len(images) >= batch_size:
                    X_train = np.array(images)
                    y_train = np.array(angles)
                    yield sklearn.utils.shuffle(X_train, y_train)
					# delete previous batch data
                    del images [:]
                    del angles [:]

# compile and train the model using the generator function
train_generator = generator(train_samples, True, batch_size)
validation_generator = generator(validation_samples, False, batch_size)

row, col, ch = 160, 320, 3
cropping_height = (70, 25)
cropping_width = (0, 0)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
# trim image to only see section with road
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
history_object = model.fit_generator(train_generator,
        steps_per_epoch = spe,
        validation_data = validation_generator,
        validation_steps = vs,
        epochs=5, verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

