import tensorflow as tf
import numpy as np
"""
The class represents single layered neural network
using softmax activation for regression
particular regression score at o/p pertains to one number from 0 to 9

This is by all means a problem of using regression for classifying digits
and not in a traditional snese a multiclass classification problem
"""
class SoftmaxRegressionOneLayerNN:
    def __init__(self, batch_size : int=100, learning_rate: float=0.001, \
                 training_epochs: int=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoches = training_epochs

    def train(self, train_images, train_labels, test_images, test_labels):
        self.tf_input = np.asarray(train_images.reshape([train_images.shape[0],784]))
        self.tf_true_pos = np.asarray(train_labels.reshape([train_labels.shape[0],10]))
        print("*****************************",self.tf_input.shape)
        print("*****************************", self.tf_true_pos.shape)
        # tf Graph Input
        x = tf.placeholder(tf.float32, [self.batch_size, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [self.batch_size, 10]) # 0-9 digits recognition => 10 classes

        # Setting weights W randomly non zero values and bias b as 0
        W = tf.Variable(tf.random_normal([784, 10], dtype=tf.float32, mean=0.0,\
                                         stddev=0.005))
        b = tf.Variable(tf.zeros([10]))

        #calculating prediction
        pred = tf.matmul(x,W) + b

        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                              (labels = y, logits = pred))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        def getNextBatch(batch_size =1, batch_count = 1):
            index = batch_size*batch_count
            return self.tf_input[index-10:index,:], self.tf_true_pos[index-10:index,:]
        # create input queues
        #train_input_queue = tf.train.slice_input_producer(
        #                            [self.tf_input, self.tf_true_pos],
        #                            shuffle=False)
        #test_input_queue = tf.train.slice_input_producer(
        #                            [test_images, test_labels],
        #                            shuffle=False, num_epochs=self.training_epoches)
        #coord = tf.train.Coordinator()

        # batch will now load up to 100 image-label-pairs on sess.run(...)
        # this is faster and also gives better result on e.g. gradient calculation
        #train_batch_images, train_batch_labels = tf.train.batch([self.tf_input, self.tf_true_pos], batch_size=self.batch_size)
        #test_batch_images, test_batch_labels = tf.train.batch([test_images, tests_labels], batch_size=10)
        print("input pipeline ready")
        with tf.Session() as sess:
            # "boilerplate" code
            sess.run([tf.local_variables_initializer(),\
            tf.global_variables_initializer()])
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(self.training_epoches):
                batch_count = 1
                epoch_cost = []
                while batch_count < (self.tf_input.shape[0])/(self.batch_size):
                    image_batch, label_batch = getNextBatch(self.batch_size, batch_count)
                    # will start reading, working data from input queue
                    # and "fetch" the results of the computation graph
                    # into raw_images and raw_labels
                    _,cost_cal = sess.run([optimizer, cost],feed_dict={ x:image_batch, y: label_batch})
                    epoch_cost.append(cost_cal)
                    batch_count += 1
                print("The Cost for epoch ", epoch," after training images ",batch_count," is ", np.mean(epoch_cost))
