"""
For loading manually downloaded data from given folder structure
project_dir
        |---data
                |---train-images.idx3-ubyte
                |---train-labels.idx1-ubyte
                |---t10k-images.idx3-ubyte
                |---t10k-labels.idx1-ubyte
** tensorflow provides hastle free means to load MNIST dataset, they can be used instead of these functions
eg : from tensorflow.examples.tutorials.mnist import input_data
"""

import os
import struct
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt




"""
takes two parameters,
1. a String indicating whether to load training or test data
2. path of the data folder

returns a tuple containing
1. images
2. labels
"""
def loadData(dataset = "training", path = "."):

    if dataset is "training":
        image_file = os.path.join(path,"train-images.idx3-ubyte")
        label_file = os.path.join(path,"train-labels.idx1-ubyte")

    elif dataset is "test":
        image_file = os.path.join(path,"t10k-images.idx3-ubyte")
        label_file = os.path.join(path,"t10k-labels.idx1-ubyte")
    else :
        ValueError, "dataset value must be either \"training\" or \"test\" "

    with open(label_file, 'rb') as lb_file :
        magic, num = struct.unpack(">II", lb_file.read(8))
        read_labels = np.fromfile(lb_file, dtype=np.uint8)
        labels = np.zeros([read_labels.shape[0],10])
        for i in range(read_labels.shape[0]):
            if read_labels[i]==1:
                labels[i,0] = 1
            elif read_labels[i]==2:
                labels[i,1] = 1
            elif read_labels[i]==3:
                labels[i,2] = 1
            elif read_labels[i]==4:
                labels[i,3] = 1
            elif read_labels[i]==5:
                labels[i,4] = 1
            elif read_labels[i]==6:
                labels[i,5] = 1
            elif read_labels[i]==7:
                labels[i,6] = 1
            elif read_labels[i]==8:
                labels[i,7] = 1
            elif read_labels[i]==9:
                labels[i,8] = 1
            elif read_labels[i]==0:
                labels[i,9] = 1

    with open(image_file, 'rb') as im_file :
        magic, num, rows, cols = struct.unpack(">IIII", im_file.read(16))
        images = np.fromfile(im_file, dtype=np.uint8).reshape(len(labels), rows, cols)

    return(images,labels)

def showRandom(images):

    fig = plt.figure()
    for i in range(1,26):
        im = images[np.random.randint(images.shape[0]),:]
        ax = fig.add_subplot(5,5,i)
        imgplot = ax.imshow(im, cmap=mpl.cm.Greys)
    plt.suptitle('Loaded Data', fontsize=16)
    plt.show()
