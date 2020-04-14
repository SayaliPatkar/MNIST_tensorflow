import os
import struct
import numpy as np
import MNISTData as mnist
import SoftmaxRegressionOneLayerNN as nnl
import CrossEntropyADAMWithCNN as nnlad

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
data_path = os.path.join(script_dir, "data")
dataset_options = ["training","test"]

#loading training data
train_images, train_labels = mnist.loadData(dataset_options[0], data_path)
#mnist.showRandom(train_images)

#loading test data
test_images, test_labels = mnist.loadData(dataset_options[1], data_path)
#mnist.showRandom(test_images)

#model = nnl.SoftmaxRegressionOneLayerNN(10,  0.0001, 100)
model = nnlad.CrossEntropyADAMWithCNN(10,  1e-4, 100)
model.train(train_images,train_labels,test_images,test_labels)
