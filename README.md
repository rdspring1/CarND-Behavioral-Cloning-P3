#**Behavioral Cloning** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/left.jpg "Left"
[image3]: ./examples/center1.jpg "Center Recovery"
[image4]: ./examples/right.jpg "Recovery Image"
[image5]: ./examples/original.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[video1]: ./track1.mp4 "Track 1 Video"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

My model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 120x320x3 RGB image   						| 
| Cropping 2D	     	| outputs 70x25x3 								|
| Convolution 5x5     	| 2x2 stride, same padding, 24 channels, 'RELU'	|
| Convolution 5x5     	| 2x2 stride, same padding, 36 channels, 'RELU'	|
| Convolution 5x5     	| 1x1 stride, same padding, 48 channels, 'RELU'	|
| Convolution 5x5     	| 1x1 stride, same padding, 64 channels, 'RELU'	|
| Fully connected		| 100 nodes, 0.1 Dropout, 'RELU'				|
| Fully connected		| 50 nodes, 0.1 Dropout, 'RELU'       			|
| Fully connected		| 10 nodes      								|
| Fully connected		| 1 nodes      									|

I based my model architecture after the solution provided by Nvidia.
[End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

I used a batch size of 32 images.
I applied dropout regularization to combat overfitting in the model. (model.py lines 81,83)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21-25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

####4. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, flipped images, and corrected left and right images. 

To capture good driving behavior, I first recorded two laps on track one using center lane driving - clockwise and counter-clockwise. 
Here is an example image of center lane driving:

![alt text][image1]

I noticed the car had trouble navigating the sharp turns in the track. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center in the sharp turns of the track.
I also used the left and right camera images by applying a correction value to steering angle. 

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then, I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles to prevent the model from leaning to one-side of the road.
For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 110,816 number of data points. 
I then preprocessed this data by cropping the image to focus only on the road and by normalizing the image channels between (-0.5, 0.5).
I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decrease and plateuing of the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Video Result

Here is my result for track1.
![alt text][video1]