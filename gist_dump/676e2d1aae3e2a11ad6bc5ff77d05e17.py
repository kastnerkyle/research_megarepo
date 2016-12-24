# Author: Kyle Kastner
# License: BSD 3-Clause
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.updates import adam
from lasagne.init import GlorotUniform
from lasagne.objectives import squared_error
from lasagne.nonlinearities import linear, rectify

from deconv import TransposeConv2DLayer, Unpool2DLayer

import numpy as np
from scipy.misc import face

from theano import tensor
import theano

random_state = np.random.RandomState(1999)
# Add batchsize, channel dim
X_train = face(gray=True)[None, None].astype('float32')
X_train = X_train / 255.
y_train = 2 * X_train
chan = X_train.shape[1]
width = X_train.shape[2]
height = X_train.shape[3]

input_var = tensor.tensor4('X')
target_var = tensor.tensor4('y')

l_input = InputLayer((None, chan, width, height), input_var=input_var)
l_conv1 = Conv2DLayer(l_input, num_filters=32, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform())
l_pool1 = MaxPool2DLayer(l_conv1, pool_size=(2, 2))

l_conv2 = Conv2DLayer(l_pool1, num_filters=32, filter_size=(1, 1),
                      nonlinearity=rectify, W=GlorotUniform())
l_depool1 = Unpool2DLayer(l_pool1, (2, 2))
l_deconv1 = TransposeConv2DLayer(l_depool1, num_filters=chan,
                                 filter_size=(3, 3),
                                 W=GlorotUniform(), nonlinearity=linear)

l_out = l_deconv1

prediction = get_output(l_out)
train_loss = squared_error(prediction, target_var)
train_loss = train_loss.mean()

valid_prediction = get_output(l_out, deterministic=True)
valid_loss = squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

params = get_all_params(l_out, trainable=True)
updates = adam(train_loss, params, learning_rate=1E-4)

train_function = theano.function([input_var, target_var], train_loss,
                                 updates=updates)
valid_function = theano.function([input_var, target_var], valid_loss)

n_epochs = 1000
for e in range(n_epochs):
    train_loss = train_function(X_train, y_train)
    valid_loss = valid_function(X_train, y_train)
    print("train: %f" % train_loss)
    print("valid %f" % valid_loss)
