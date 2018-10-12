import time
import numpy as np
import tensorflow as tf

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class MNIST(object):
    def __init__(self):
        # Create the model
        self.model_name = '/home/elfeki/Workspace/DCGAN/mnist/data/mnist.ckpt'
        self.x = tf.placeholder(tf.float32, [None, 784])
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def deepnn(self, x):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        return y_conv

    def build_model(self, mnist):
        # Build the graph for the deep net
        y_conv = self.deepnn(self.x)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        params = tf.trainable_variables()
        train_step = tf.train.AdamOptimizer(
            1e-4).minimize(cross_entropy, var_list=params)

        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        start_tic = time.clock()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in range(20000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' %
                          (i, train_accuracy))
                train_step.run(
                    feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            print('test accuracy %g' % accuracy.eval(feed_dict={
                self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))

            saver.save(sess, self.model_name)
        end_toc = time.clock()
        print('Time for training: {}'.format(end_toc - start_tic))

    def classify(self, inputs):
        y_conv = self.deepnn(self.x)
        sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(sess, self.model_name)
        # from IPython import embed; embed()
        y_pred = sess.run(y_conv, feed_dict={
            self.x: inputs, self.keep_prob: 1.0})
        y = np.argmax(y_pred, 1)
        return y


def get_dist(p, n):
    pk = np.zeros(n)
    for x in p:
        pk[x] += 1
    for i in range(n):
        pk[i] = pk[i] * 1.0 / len(p)
    return pk
