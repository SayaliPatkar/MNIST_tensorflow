import tensorflow as tf
import numpy as np
"""
The class represents single layered neural network
using softmax activation for regression
particular regression score at o/p pertains to one number from 0 to 9

For training gradient descent optimization is used with customized batching logic
Loss calculations are done using softmax cross entropy
"""
class SoftmaxRegressionOneLayerNN:
    def __init__(self, batch_size : int=100, learning_rate: float=0.001, \
                 training_epochs: int=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoches = training_epochs

    def train(self, train_images, train_labels, test_images, test_labels):

        # mnist data image of shape 28*28=784
        self._input = np.asarray(train_images.reshape([train_images.shape[0],784]))
        # 0-9 digits recognition => 10 classes
        self._true_pos = np.asarray(train_labels.reshape([train_labels.shape[0],10]))

        self._test_input = np.asarray(test_images.reshape([test_images.shape[0],784]))
        self._test_lab = np.asarray(test_labels.reshape([test_labels.shape[0],10]))

        # tf Graph Input
        x = tf.placeholder(tf.float32, [self.batch_size, 784])
        y = tf.placeholder(tf.float32, [self.batch_size, 10])

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

        def getNextBatch(batch_size =1, batch_count = 1, type = "train"):
            index = batch_size*batch_count
            if type is "train":
                return self._input[index-10:index,:], self._true_pos[index-10:index,:]
            else:
                return self._test_input[index-10:index,:], self._test_lab[index-10:index,:]

        with tf.Session() as sess:
            # "boilerplate" code
            sess.run([tf.local_variables_initializer(),\
            tf.global_variables_initializer()])
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            test_batch_count1 = 1
            total_accuracy1 = []
            while test_batch_count1 < (self._test_input.shape[0])/(self.batch_size):
                image_test_batch, label_test_batch = getNextBatch(self.batch_size, test_batch_count1, "test")
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                total_accuracy1.append(accuracy.eval({x: image_test_batch , y: label_test_batch}))
                test_batch_count1 += 1
            total_accuracy1 = np.mean(total_accuracy1)
            print ("Accuracy:", total_accuracy1)

            for epoch in range(self.training_epoches):
                if epoch is 0:
                    print("Starting Optimization...")
                batch_count = 1
                epoch_cost = []
                while batch_count < (self._input.shape[0])/(self.batch_size):
                    image_batch, label_batch = getNextBatch(self.batch_size, batch_count, "train")
                    # will start reading, working data from input queue
                    # and "fetch" the results of the computation graph
                    # into raw_images and raw_labels
                    _,cost_cal = sess.run([optimizer, cost],feed_dict={ x:image_batch, y: label_batch})
                    epoch_cost.append(cost_cal)
                    batch_count += 1
                print("The Cost for epoch ", epoch," after training images ",batch_count," is ", np.mean(epoch_cost))
            print("Optimization Complete, now evaluating accuracy...")
            # Test model
            test_batch_count = 1
            total_accuracy = []
            while test_batch_count < (self._test_input.shape[0])/(self.batch_size):
                image_test_batch, label_test_batch = getNextBatch(self.batch_size, test_batch_count, "test")
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                total_accuracy.append(accuracy.eval({x: image_test_batch , y: label_test_batch}))
                test_batch_count += 1
            total_accuracy = np.mean(total_accuracy)
            print ("Accuracy:", total_accuracy)
