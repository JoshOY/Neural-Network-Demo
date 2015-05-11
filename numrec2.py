#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

"""
固定参数
"""
PNG_WIDTH_HEIGHT = 2072
NUM_GRID_LENGTH = 28

"""
可调参数
"""
# 训练集数目，为40时使用左上角40 * 40个样本作为训练样本
train_set_size = 70
# 测试集数目，为10时使用左上角10 * 10个样本作为测试样本
test_set_size = 20
# 神经网络层级
NN_LAYER = [NUM_GRID_LENGTH * NUM_GRID_LENGTH, 400, 150, 10]
# Learning rate (alpha)
param_learning_rate = 0.11
# Epochs
param_epochs = 100000

"""
全局变量
"""
data_vec = []
test_vec = []
nn = None


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=param_learning_rate, epochs=param_epochs):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0:
                print 'epochs:', k

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


def getnumc(fn, digits=train_set_size * NUM_GRID_LENGTH):
    """ 返回数字特征（灰度） """
    img = cv2.imread(fn)
    xtz = np.zeros((digits, digits))

    for now_h in xrange(0, digits):
        for now_w in xrange(0, digits):
            b = img[now_h, now_w, 0]
            g = img[now_h, now_w, 1]
            r = img[now_h, now_w, 2]
            btz = 255 - b
            gtz = 255 - g
            rtz = 255 - r
            if btz > 0 or gtz > 0 or rtz > 0:
                nowtz = 0
            else:
                nowtz = 1
            xtz[now_w, now_h] = nowtz
    return xtz


def init():
    global data_vec
    for num in range(10):
        print 'Loading jpg file of number ' + str(num) + '...'
        jpg_name = './train/mnist_train' + str(num) + '.jpg'
        data_vec.append(getnumc(jpg_name))
    with open('./pickles/datavec.pkl', 'wb') as pklfile:
        pickle.dump(data_vec, pklfile)
    print 'Dump done...'


def init_easy():
    global data_vec
    with open('./pickles/datavec.pkl', 'rb') as pklfile:
        data_vec = pickle.load(pklfile)
    print len(data_vec)
    print data_vec[0].shape


def nn_train():
    global nn
    nn = NeuralNetwork(NN_LAYER, activation='sigmoid')
    train_X = []
    train_y = []
    for num in range(10):
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y[num] = 1
        y = np.array(y)
        print 'y = ' + str(y)
        print 'Now training samples of number ' + str(num) + '...'
        print data_vec[num][0:0 + NUM_GRID_LENGTH,
                      0:0 + NUM_GRID_LENGTH]
        for train_index_x in range(train_set_size):
            for train_index_y in range(train_set_size):
                print '#%d: (x0, y0) to (x1, y1): ((%d, %d), (%d, %d))' % (train_index_x * train_set_size + train_index_y,
                                                                           train_index_x * NUM_GRID_LENGTH,
                                                                           train_index_y * NUM_GRID_LENGTH,
                                                                           train_index_x * NUM_GRID_LENGTH + 27,
                                                                           train_index_y * NUM_GRID_LENGTH + 27)
                X_append = data_vec[num][(train_index_x * NUM_GRID_LENGTH):(train_index_x * NUM_GRID_LENGTH + NUM_GRID_LENGTH),
                               (train_index_y * NUM_GRID_LENGTH):(train_index_y * NUM_GRID_LENGTH + NUM_GRID_LENGTH)]\
                    .reshape(1, NUM_GRID_LENGTH * NUM_GRID_LENGTH)[0]
                train_X.append(X_append)
                train_y.append(y.copy())
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    print train_X
    print train_y
    nn.fit(train_X, train_y)

    print 'Training done!'
    with open('./pickles/nn.pkl', 'wb') as pklfile:
        pickle.dump(nn, pklfile)
    print 'Dump nn done.'


def test_init():
    global test_vec
    for num in range(10):
        print 'Loading jpg file of number ' + str(num) + '...'
        jpg_name = './test/mnist_test' + str(num) + '.jpg'
        test_vec.append(getnumc(jpg_name, digits=test_set_size * NUM_GRID_LENGTH))
    with open('./pickles/testvec.pkl', 'wb') as pklfile:
        pickle.dump(test_vec, pklfile)
    print 'Dump done...'


def test_init_easy():
    global test_vec
    global nn
    with open('./pickles/testvec.pkl', 'rb') as pklfile:
        test_vec = pickle.load(pklfile)
    with open('./pickles/nn.pkl', 'rb') as pklfile:
        nn = pickle.load(pklfile)

def test():
    correct_sum = 0
    sum = 0
    for num in range(10):
        print 'Now testing samples of number ' + str(num) + '...'
        print test_vec[num][0:0 + NUM_GRID_LENGTH,
                      0:0 + NUM_GRID_LENGTH]
        for test_index_x in range(test_set_size):
            for test_index_y in range(test_set_size):
                print '#%d: (x0, y0) to (x1, y1): ((%d, %d), (%d, %d))' % (test_index_x * train_set_size + test_index_y,
                                                                           test_index_x * NUM_GRID_LENGTH,
                                                                           test_index_y * NUM_GRID_LENGTH,
                                                                           test_index_x * NUM_GRID_LENGTH + 27,
                                                                           test_index_y * NUM_GRID_LENGTH + 27)
                test_append = test_vec[num][(test_index_x * NUM_GRID_LENGTH):(test_index_x * NUM_GRID_LENGTH + NUM_GRID_LENGTH),
                               (test_index_y * NUM_GRID_LENGTH):(test_index_y * NUM_GRID_LENGTH + NUM_GRID_LENGTH)]\
                    .reshape(1, NUM_GRID_LENGTH * NUM_GRID_LENGTH)[0]
                predict = nn.predict(test_append)
                pred_num = 0
                pred_percent_max = -99
                for index in xrange(len(predict)):
                    if predict[index] > pred_percent_max:
                        pred_num = index
                        pred_percent_max = predict[index]
                    else:
                        pass
                print "Predicted number: " + str(pred_num)
                sum += 1
                if pred_num == num:
                    print 'Correct!'
                    correct_sum += 1
                else:
                    print 'Incorrect!'
    print 'Accuracy: ' + str(float(correct_sum) / float(sum))


if __name__ == '__main__':
    init()
    init_easy()
    nn_train()
    test_init()
    test_init_easy()
    test()
