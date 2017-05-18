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


def add_datum_to_collection(images, image, measurements, measurement):
    # add to collection
    images.append(image)
    measurements.append(measurement)
    # add flipped image
    images.append(cv2.flip(image, 1))
    measurements.append(measurement * -1.0)


#load training set to memory (X - images, Y - steering angle)
'''
for line in lines:
    #center camera
    relative_img_file = get_relative_path(line[0])
    image = cv2.imread(relative_img_file)
    measurement = float(line[3])
    add_datum_to_collection(images, image, measurements, measurement)

    #left
    relative_img_file = get_relative_path(line[1])
    image = cv2.imread(relative_img_file)
    add_datum_to_collection(images, image, measurements, measurement + 0.25)

    #right
    relative_img_file = get_relative_path(line[2])
    image = cv2.imread(relative_img_file)
    add_datum_to_collection(images, image, measurements, measurement - 0.25)
'''
#wrap data in numpy array
#X_train = np.array(images)
#Y_train = np.array(measurements)

#print("input shapes:")
#print(X_train.shape)
#print(Y_train.shape)


import keras.models
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout


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

#model = keras.models.load_model('model.h5')

#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=20)

model.fit_generator(train_generator,
                    steps_per_epoch= (len(train_samples)) / BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps= (len(validation_samples)) / BATCH_SIZE,
                    epochs=10)

model.save('model.h5')
