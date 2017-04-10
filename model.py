import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open("../data/course1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split("\\")[-1]
    current_path = "../data/course1/IMG/" + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile('adam', 'mse')
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
model.save('model.h5')
