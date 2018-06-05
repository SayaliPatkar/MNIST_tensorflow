import tensorflow as tf

"""
The class represents single layered neural network
using softmax activation for regression
particular regression score at o/p pertains to one number from 0 to 9

This is by all means a problem of using regression for classifying digits
and not in a traditional snese a multiclass classification problem
"""
class SoftmaxRegressionOneLayerNN:
    def __init__(self, batch_size : int=100, learning_rate: float=0.001, \
                 training_epochs: int= 10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoches = training_epochs

    def train(train_images, train_labels, test_images, test_labels):

        self.input = train_images
        self.true_output = train_labels
        self.sample_count =

        # Setting weights W randomly non zero values and bias b as 0
        W = tf.Variable(tf.random_normal([784, 10], dtype=tf.float62, mean=0.0,\
                                         stddev=0.005))
        b = tf.Variable(tf.zeros([10]))

        #calculating prediction
        self.output = tf.matmul(self.input,W) + b

        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                              (labels = self.true_output, logits = self.output))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

    def validate(test_images, test_labels):
