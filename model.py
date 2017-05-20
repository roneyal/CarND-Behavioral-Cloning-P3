import csv
import cv2 #open CV

import generator
import random

log_file = '../data/driving_log.csv'
relative_img_path = '../data/IMG/'

BATCH_SIZE = 32

#get relative file path from full path
def get_relative_path(full_path):
    filename = full_path.split('/')[-1]
    relative_path = relative_img_path + filename
    return relative_path

#load the entire CSV file
#CSV structure: Center Image, Left Image, Right Image, Steering Angle, Throttle, Break, Speed
lines = []
with open(log_file) as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        angle = float(line[3])
        if angle > 0.01 or angle < -0.01 or random.uniform(0.0,1.0) < 0.3:
            lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator.generator(validation_samples, batch_size=BATCH_SIZE)

images = []
measurements = []

import keras.models
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.utils import plot_model

def create_model():

    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    # preprocessing layer (normalization)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # Convolutional layer 1
    model.add(Conv2D(24, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolutional layer 2
    model.add(Conv2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolutional layer 3
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Convolutional Layer 4
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(74, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

model = create_model()

plot_model(model, to_file='model.png')

model.fit_generator(train_generator,
                    steps_per_epoch= (len(train_samples)) / BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps= (len(validation_samples)) / BATCH_SIZE,
                    epochs=10)

model.save('model.h5')
