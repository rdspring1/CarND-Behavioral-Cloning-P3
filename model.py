import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split

samples = []
with open("../data/course1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# TODO Add Multiple Cameras
# TODO Recovery Laps

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split("\\")[-1]
                name = "../data/course1/IMG/" + filename
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle
                images.append(center_image)
                images.append(image_flipped)
                angles.append(center_angle)
                angles.append(angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train) 

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile('adam', 'mse')
history_object = model.fit_generator(train_generator,
        steps_per_epoch = len(train_samples),
        validation_data = validation_generator,
        validation_steps = len(validation_samples), 
        epochs=10, verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

