import sklearn
import random
import numpy as np
import cv2

relative_img_path = '../data/IMG/'

def add_datum_to_collection(images, image, measurements, measurement):
    #if random.uniform(0.0, 1.0) < 0.5:
    # add to collection
    images.append(image)
    measurements.append(measurement)
    #else:
    # add flipped image
    images.append(cv2.flip(image, 1))
    measurements.append(measurement * -1.0)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                name = relative_img_path + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                name = relative_img_path + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = center_angle + 0.25

                name = relative_img_path + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = center_angle - 0.25


                rand = random.uniform(0.0,1.0)
                #if rand < 0.4:
                add_datum_to_collection(images, center_image, angles, center_angle)
                #elif rand < 0.7:
                add_datum_to_collection(images, left_image, angles, left_angle)
                #else:
                add_datum_to_collection(images, right_image, angles, right_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)