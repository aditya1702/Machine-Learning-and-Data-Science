import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import rnn
from decorators import define_scope

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


class Model:

    def __init__(self, data, label, weights, biases):
        self.data = data
        self.label = label
        self.weights = weights
        self.biases = biases
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        data_unstacked = tf.unstack(self.data, n_steps, 1)
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, data_unstacked, dtype=tf.float32)
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    @define_scope
    def optimize(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                      labels=self.label))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return cost, optimizer.minimize(cost)

    @define_scope
    def error(self):
        correct_pred = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    image = tf.placeholder("float", [None, n_steps, n_input])
    label = tf.placeholder("float", [None, n_classes])
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    model = Model(image, label, weights, biases)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(model.optimize, feed_dict={image: batch_x, label: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(model.error, feed_dict={image: batch_x, label: batch_y})
            # Calculate batch loss
            loss, _ = sess.run(model.optimize, feed_dict={image: batch_x, label: batch_y})
            print "Iter " + str(step*batch_size) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", \
        sess.run(model.error, feed_dict={image: test_data, label: test_label})

main()
