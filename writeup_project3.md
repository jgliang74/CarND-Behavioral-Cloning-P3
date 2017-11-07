
## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/model.png "Model Visualization"
[image2]: ./writeup_images/center_lane_driving.jpg "Center lane driving"
[image3]: ./writeup_images/normal_image.jpg "Normal Image"
[image4]: ./writeup_images/flipped_image.jpg "Flipped Image"

#### Files Submitted & Code Quality

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_project3.md
* video.mp4 a video recording of the vehicle driving autonomously
* bclone_model.html saved version of bclone_model.ipynb used to do model training for resubmission
* model_resubmit.h5 containing a trained neural nework with dropout and early stop
* video_resubmit.mp4 a video recording of the vehicle driving autonomouly with resubmitted model

##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Model Architecture and Training Strategy

##### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 67-82) 

The model includes RELU layers to introduce nonlinearity (code line 73-77), and the data is normalized in the model using a Keras lambda layer (code line 69-71). 

##### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54-55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85)

##### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

##### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with various known architectures starting with simple ones.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because I sucessfully used it in the previous project to recognize German traffic signs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I started augumenting the training data set by simply adding horizontally flipped images.

Then I also added in drop out and max pooling layers to further alleviate data overfiting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially in the sharp turns after the bridge. To improve the driving behavior in these cases, I aslo tried to collect training data in the recovery mode (purposely drove off the lane, and collected data when driving back to the center), but it was not successful. Therefore, I decided to go for more powerful model (NVDIA model introduced in the lecture) 

##### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the 5 convolutional layers and 4 fully connected layers.

Here is a visualization of the architecture 

![Model Architecture][image1]

##### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![example image of center lane driving][image2]

I then tried to record the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to recover from the off lane center postion during autonuous mode. With the recovery data, the car drove better in the autonuous mode, however, it still failed in certain turns ocassionally. This leads me to consider using left camera and right camera as weel. Later I found this is most efficient way to generate enough training data. With all images captured by three cameras, I can only record one lap center land driving data to make car nicely performing autonuous mode driving.

To augment the data set, I also flipped images and angles thinking that this would provide balanced training data for left turns and right turns. For example, here is an image that has then been flipped:

![Normal Image][image3]
![Flipped Image][image4]

After the collection process, I had 15582 number of data points (6x times more than original images). I then preprocessed this data by normalizing and mean centering the data. This is done by adding lamda layer in the model.

To speed up the training, the images were also cropped down to smaller sizes which still maintain useful information. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Early stop and Model check feature have been implemented during model training. The total number of epoch was set to10, however the training will be terminated if no progress in reducing val_loss for more than two consecutive steps (patience = 2). The model with least val_loss will be saved during training by setting save_best_only=True in ModelCheckpoint callback. Refer to bclone_model.html for the details on training progress.

