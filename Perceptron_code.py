#import numpy as np
import math
import random

x_data = [[0., 0.],
          [10., 0.],
          [0., 10.],
          [10., 10.]]

y_data = [1., 1., 0., 0.]



def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))


def sigmoid_derivate(o):
    return o * (1.0 - o)

def train(x_data, y_data):

    w0, w1, w2 = [0.1, -1.23, 0.98]
    lr = 0.1
    epochs = 1000000

    print("Training...")

    for _ in range(epochs):

        w0_d = []
        w1_d = []
        w2_d = []

        for data, label in zip(x_data, y_data):

            o = sigmoid(w0*1.0 + w1*data[0] + w2*data[1])
            error = 2.*(o - label) * sigmoid_derivate(o)

            w0_d.append(error * 1.0)
            w1_d.append(error * data[0])
            w2_d.append(error * data[1])

        w0 = w0 - sum(w0_d) * lr
        w1 = w1 - sum(w1_d) * lr
        w2 = w2 - sum(w2_d) * lr
#---------------------------------------------------------------------

    x_data = [[5., 5.],
          [100., 0.],
          [0., 5.],
          [10., 10.]]

    for data in x_data:
        o = sigmoid(w0*1.0 + w1*data[0] + w2*data[1])
        print(o)
        print("-----------------------")


train(x_data, y_data)