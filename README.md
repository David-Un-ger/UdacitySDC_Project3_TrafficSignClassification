# **Project 3: Traffic Sign Recognition** 

## Writeup / Readme


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/normalization.png "Normalization"
[image3]: ./examples/example_signs.png "Traffic Signs"
[image4]: ./examples/own_images.png "Traffic Signs"
[image5]: ./examples/top5.png "Traffic Signs - TOP5 Predictions"



![alt text][image3]

---


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Task 1: Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one. 

You're reading it! 

#### 2. Here is a link to my [project code](https://github.com/FinnHendricks/UdacitySDC_Project3_TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb)

---
### Task 2: Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is __34799__ samples.
* The size of the validation set is __4410__ samples.
* The size of test set is __12630__ samples.
* The shape of a traffic sign image is __32x32x3__.
* The number of unique classes/labels in the data set is __43__.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is divided per class. One can see that e.g. class 0 or class 19 have very few samles, whereas class 1 or 2 have by a factor of 10 more samples. Note that the y axis is scaled logarithmic to visualize also the classes with very few samples.

![alt text][image1]

---
### Task 3: Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

The only preprocessing which I used is normalization. The proposed normalization, subtracting 128 and dividing then by 128 works quite well.

Grayscaling was not used because if you reflect the problem, the colors can help the neural network to classify the images correctly. Some images are blue and others red, so this helps the CNN understand the data.

After training the CNN, one can see that there is a very strong overfitting. In general more training data is very useful to improve the accuracy. But also without new data, the required performance could be achieved, why no further training data was created.
Methods for additional data would be e.g.:
* Slight rotation  
* Slight zoom
* Slight shift of the image in x or/and y
* Vertical flip for some images (e.g. caution or bumpy road)
* Horizontal flip for some images (e.g. No vehicles or no entry)
* Add/subtract random noise

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Number		| Layer         						|     Description	        					| 
|---------------|---------------------------------------|-----------------------------------------------| 
|				| Input         						| 32x32x3 RGB image   							| 
|conv1			| Convolution 3x3     					| 1x1 stride, same padding, outputs 32x32x8 	|
|				| RELU									|												|
|conv2			| Convolution 3x3     					| 1x1 stride, same padding, outputs 32x32x8 	|
|				| RELU									|												|
|				| Max pooling	      					| 2x2 stride,  outputs 16x16x8 				    |
|conv3			| Convolution 3x3     					| 1x1 stride, same padding, outputs 16x16x16 	|
|				| RELU									|												|
|conv4			| Convolution 3x3     					| 1x1 stride, same padding, outputs 16x16x16 	|
|				| RELU									|												|
|				| Max pooling	      					| 2x2 stride,  outputs 8x8x16 				    |
|				| Flatten			    				| pooled conv4, outputs 1024			 		|
|				| Flatten			    				| pooled conv2, outputs 2048					|
| 				| Concatenate							| conv2 and conv4, outputs 3072        			|
|fc1			| Fully connected						| outputs 1024 									|
|				| RELU									|         										|
|fc2			| Fully connected						| outputs 256									|
|				| RELU									|												|
|fc3			| Fully connected						| outputs 43									|
|				| Softmax								|												|
 
Having 2 convolution 3x3 layers and then a pooling layer is quite similar to having a single 5x5 layer and then a pooling layer, but the first one is more nonlinear, why it is benefitial in this task.

Concatenating the pooled conv4 layer with the pooled conv2 layer brought an additional accuracy improvement of about 2 or 3 %.

Using more channels would be a little bit better for the accuracy performance, but the overfitting would be also a lot higher.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the following parameters:
* Optimizer: Adam
* Epochs: 50
* Batch Size: 2000
* Learning Rate: 0.002
* Dropout during training: 60%

The Adam optimizer trains better than SGD and showed good results. 
The quality of about 92% is reached after ~20 Epochs. After ~50 Epochs, the model reaches the maximum accuracy and starts overfitting. Therefore, the training is stopped after 50 Epochs.
Using dropout shows good results. The model is still underfitting, but less worth than without using dropout.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of __100%__
* validation set accuracy of __94.1%__ 
* test set accuracy of __94.4%__



* __What was the first architecture that was tried and why was it chosen?__

As a basis, the LeNet architecture was a good starting point. 


* __What were some problems with the initial architecture?__

Compared to other CNNs, the LeNet is very small with very few weigths.


* __How was the architecture adjusted and why was it adjusted?__ 

The main adjustment was giving the CNN more channels. 

Further, the conv5 layers where changed to two conv3 layers. In addition, not only the final pooling layer was flattened and connected to the dense layers, but both pooling layers were directly connected to the dense layers.


* __Which parameters were tuned? How were they adjusted and why?__

In general almost all parameters where adjusted. Some examples:

o Number of channels were increased: more parameters give the CNN more flexibility to learn which results in better results

o Padding for conv layers where changed to "SAME": For the traffic sign images there is a lot if information at the sides of the images, which can be preserved if the padding doesn´t cut information from the sides.

o Size of the dense layers: The size was increased, because LeNet was only created for classifying 10 classes. More classes need bigger dense layers.

---
### Task 4: Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. 


Here are 20 German traffic signs that I found on the web:

![alt text][image4]

In general, the images should be easy to classify, because the lightning conditions and the contrast for all images is very good. 
On problem could be, that most of the images are synthetic, so the CNN has never seen something like this before.

Problematic could be image 6 (general caution), because there is an additional frog image below and image 10 (no entry), because there are drawings on the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction for the first 5 images. All predictions are stated above the images itself:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 kph	      		| 30 kph							| 
| 20 kph     			| 20 kph 										|
| 30 kph				| 30 kph										|
| 50 kph	      		| 30 kph						 				|
| General caution		| General caution      							|



For all 20 images, the traffic sign classifier reached an accuracy of 70%, which means that there are 6 missmatches in total.

Missmatch 1: 100kph detected as 30kph: To read numbers, many training samples are needed. For differntiating between the numbers, the CNN needs more training.

Missmatch 2: 50kph detected as 30kph: Same problem as missmatch 1

Missmatch 3: General caution detected as bicycles crossing: The traffic sign is off center and there is this additional frog sign. Both make it hard for the CNN to detect the correct answer.

Missmatch 4: Bumpy road detected as bicycles crossing: Both signs look quite similar, but with more training (data) it should be possible to detect this sign correctly.

Missmatch 5: Road work detected as Stop: Both signs don´t really look similar. Maybe the CNN has a problem because the sign is yellow, which is probably not the case in the training set. 
 
Missmatch 6: Beware of ice was detected as children crossing: Both signs look quite similar, but with more training (data) it should be possible to detect this sign correctly.



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

The top 5 predictions of the 5 images is above the images below.

Images 1, 2 and 5 are all predicted correctly and with 100% certainty.

Image 3 (beware of snow/ice) was misclassified, because the CNN is stating that 70% childres crossing is detected. The Top2 is with 15% certainty the correct prediction of ice/snow.

Image 4 was classified correctly as no passing. The certainty is only 86% because the "end of no passing" has also 14%. Both signs look really similar, so this result makes sense.

![alt text][image5]
