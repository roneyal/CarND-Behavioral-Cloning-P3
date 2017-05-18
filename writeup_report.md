
**Behavioral Cloning Project Report**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_05_14_19_57_39_609.jpg "center"
[image3]: ./examples/center_2017_05_18_20_38_29_292.jpg "Recovery Image"
[image4]: ./examples/center_2017_05_18_20_38_29_373.jpg "Recovery Image"
[image5]: ./examples/center_2017_05_18_20_38_29_460.jpg "Recovery Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* generator.py containing helper code for loading files and the generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 - a video showing autonomous driving

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with four convolutional layers, followed by three fully connected layers. 
It is defined in the function create_model (line 42 in model.py).
The data is passed through the network after cropping and normalization. I used a relu activation function between the layers.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 62, 64). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. I noticed that the validation loss was similar to the training loss.
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and utilization of all three cameras.
The data set was imbalanced due to the fact that most of the time the steering angle was zero, so I subsampled the images with very small steering angles (I used only 20% of them).

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to combine convolutional layers with fully connected layers, as in many other successful architectures.
I gradually added more parameters to the network to allow better fitting to the training data, until I was satisfied with the results.

To combat the overfitting, I added dropout and made sure the validation loss was close to the training loss.

Then I gradually added more training data until the model was successful at driving around the track.

####2. Final Model Architecture

The final model architecture (model.py lines 42-66) consisted of a convolution neural network with the following layers:

1. Cropping 70 pixels from the top and 25 from the bottom
2. Normalization
3. Three 5x5 convolutional layers with increasing depth
4. Another 3x3 convolutional layer
5. Three fully connected layers, the first two layers with dropout (keep probability of 75%)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from bad situations in which a cumulative error caused it to drift to the side. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles (generator.py line 15) and used images from the left and right cameras (offsetting the steering angle accordingly, generator.py lines 35 and 39) 

After the collection process, I had over 12000 raw data points, each containing images from three cameras. I subsampled the data - taking only 20% of the data points with a very small steering angle (between -0.01 and +0.01), and for each point I randomly selected the center image, left image or right image, with a 50% chance of flipping the image (see lines 9-16 and 42-48 of generator.py)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10, as evidenced by the fact that the loss stopped decreasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
