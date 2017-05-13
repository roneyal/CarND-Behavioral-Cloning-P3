import csv
import cv2 #open CV
import numpy as np

log_file = '../data/driving_log.csv'
relative_img_path = '../data/IMG/'

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
        lines.append(line)

images = []
measurements = []

#load training set to memory (X - images, Y - steering angle)
for line in lines:
    relative_img_file = get_relative_path(line[0])
    image = cv2.imread(relative_img_file)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

#wrap data in numpy array
X_train = np.array(images)
Y_train = np.array(measurements)

print("input shapes:")
print(X_train.shape)
print(Y_train.shape)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D

model = Sequential()
#preprocessing layer (normalization)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#Convolutional layer 1
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Convolutional layer 2
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#Fully connected layers
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')