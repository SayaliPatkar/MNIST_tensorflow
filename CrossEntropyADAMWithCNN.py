import tensorflow as tf
import numpy as np
"""
The class represents convolutional neural network
using softmax activation for regression
particular regression score at o/p pertains to one number from 0 to 9

For training ADAM optimization is used with customized batching logic
Loss calculations are done using softmax cross entropy
"""
class CrossEntropyADAMWithCNN:
    def __init__(self, batch_size : int=100, learning_rate: float=1e-4, \
                 training_epochs: int=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoches = training_epochs

    def train(self, train_images, train_labels, test_images, test_labels):

        # mnist data image of shape 28*28=784
        self._input = np.asarray(train_images.reshape([train_images.shape[0],28,28,1]), dtype=np.float32)
        # 0-9 digits recognition => 10 classes
        self._true_pos = np.asarray(train_labels.reshape([train_labels.shape[0],10]))

        self._test_input = np.asarray(test_images.reshape([test_images.shape[0],28,28,1]))
        self._test_lab = np.asarray(test_labels.reshape([test_labels.shape[0],10]))

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # strides=[1, 2, 2, 1] / strides=[1, 1, 1, 1]
        # for stride [x,W,H,x] only W and H dimensions imaprt meaning
        # W horizontal stride , H vertical stride
        # x values will always be 1
        # padding='SAME' output tensor will have the same dim as i/p sensor
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        # tf Graph Input
        x = tf.placeholder(tf.float32, [self.batch_size, 28,28,1])
        y = tf.placeholder(tf.float32, [self.batch_size, 10])
        """
        *************************************************************Layer 1 starts conv followed by max pooling*******************************************
        Formula (n+ 2*p - f)/s  + 1
        28*28 image converted to     : (28 + 2*0 -5)/1 + 1 = 24   Hence o/p 24*24
        padding='SAME' output tensor will have the same dim as i/p sensor without padding o/p will be 24*24
        because of padding same , 28*28 instead of 24*24
        max pooling 2*2 and stride 2 : (28 + 2*0 -2)/2 + 1 =14   Hence o/p 14*14
        """
        # 32 5*5 filters are applied in first convolutional layer, 1 coreesponds
        # to input channels since MNIST images are monochrome it is 1
        # in case of RGB images it will be 3
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])


        # Convolutional layers then typically apply a ReLU activation function
        # to the output to introduce nonlinearities into the model
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        # 2*2 filter with stride 2
        h_pool1 = max_pool_2x2(h_conv1)
        """
        *************************************Layer 2 starts conv followed by max pooling*******************************************
        Formula (n+ 2*p - f)/s  + 1
        12*12 image converted to     : (14 + 2*0 -5)/1 + 1 = 10   Hence o/p 10*10
        padding='SAME' output tensor will have the same dim as i/p sensor without padding o/p will be 10*10
        because of padding same , 14*14 instead of 10*10
        max pooling 2*2 and stride 2 : (14  + 2*0 -2)/2 + 1 = 7   Hence o/p 7*7
        """
        # 64 5*5 filters are applied in first convolutional layer, 32 coreesponds
        # to output channels from previous layer which is 32
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # 2*2 filter with stride 2
        h_pool2 = max_pool_2x2(h_conv2)
        """
        *****************************************Layer 3 starts fully connected layer*******************************************
        fully-connected layer with 1024 neurons
        7*7*64 = after flattening the o/p of layer 2
        """
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        """
        *************************************************************Layer 3 Dropout *******************************************
        """
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        """
        *****************************************Layer 4 starts fully connected layer*******************************************
        fully-connected layer with 10 neurons
        1024 = after flattening the o/p of layer 2
        """
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                              (labels = y, logits = pred))

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                total_accuracy1.append(accuracy.eval({x: image_test_batch , y: label_test_batch, keep_prob: 0.5}))
                test_batch_count1 += 1
            total_accuracy1 = np.mean(total_accuracy1)*100
            print ("Accuracy:", total_accuracy1, " percent")

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
                    _,cost_cal = sess.run([optimizer, cost],feed_dict={ x:image_batch, y: label_batch, keep_prob: 0.5})
                    epoch_cost.append(cost_cal)
                    batch_count += 1
                print("The Cost for epoch ", epoch," after training images ",batch_count," is ", np.mean(epoch_cost))
            print("Optimization Complete, now evaluating accuracy...")
            # Test model
            test_batch_count = 1
            total_accuracy = []
            while test_batch_count < (self._test_input.shape[0])/(self.batch_size):
                image_test_batch, label_test_batch = getNextBatch(self.batch_size, test_batch_count, "test")
                total_accuracy.append(accuracy.eval({x: image_test_batch , y: label_test_batch, keep_prob: 0.5}))
                test_batch_count += 1
            total_accuracy = np.mean(total_accuracy1)*100
            print ("Accuracy:", total_accuracy, " percent")
