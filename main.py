import os
import struct
import numpy as np
import MNISTData as mnist

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
data_path = os.path.join(script_dir, "data")
dataset_options = ["training","test"]

#loading training data
train_images, train_labels = mnist.loadData(dataset_options[0], data_path)
mnist.showRandom(train_images)

#loading test data
test_images, test_labels = mnist.loadData(dataset_options[1], data_path)
#mnist.showRandom(test_images)
