The following code is written to classify MNIST dataset images

Dataset :

MNIST dataset uses 28*28-pixel greyscale images and label information stored in the form of IDX files.The logic for loading data manually can be found in MNISTData.py. 

For this project MNIST data set is used that can be downloaded from here
http://yann.lecun.com/exdb/mnist/

Note : tensorflow also provides hastle free means to load MNIST dataset, they can be used instead of these functions eg : from tensorflow.examples.tutorials.mnist import input_data

You can find two training approaches that are used for training

1. SoftmaxRegressionOneLayerNN.py
The class represents single layered neural network and uses softmax activation for regression. 
The regression score at o/p pertains to one number from 0 to 9
For training gradient descent optimization is used with customized batching logic
Loss calculations are done using softmax cross entropy

Accuracy achieved after training : 90%

2. CrossEntropyADAMWithCNN.py
The class represents convolutional neural network using two convolutional layer followed by two fully connected layers
Dropout for is also imlemented after Fully Connected Layer 1
I have tried to explain all the CNN computations and output sizes in the comments
For training ADAM optimization is used with customized batching logic
Loss calculations are done using softmax cross entropy

Accuracy achieved after training : 99.2%
