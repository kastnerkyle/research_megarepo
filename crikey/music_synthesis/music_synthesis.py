# Authr: Kyle Kastner
# License: BSD 3-Clause
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.updates import adam
from lasagne.init import GlorotUniform
from lasagne.objectives import squared_error
from lasagne.nonlinearities import linear, rectify
from deconv import TransposeConv2DLayer, Unpool2DLayer

import pickle
import numpy as np

import os
from theano import tensor
import theano
import sys
from kdllib import fetch_fruitspeech_spectrogram, run_loop, list_iterator

speech = fetch_fruitspeech_spectrogram()
data = speech["data"]
# 10 classes
X_train = data[0] / 10.
X_train = X_train[None].astype("float32")
y_train = X_train.astype("float32")
minibatch_size = 1

# Make easy iterators
data = [(X_train),]
target = [(y_train),]
train_itr = list_iterator([data, target], minibatch_size, axis=0, stop_index=1)
valid_itr = list_iterator([data, target], minibatch_size, axis=0, stop_index=1)
X_mb, y_mb = train_itr.next()
train_itr.reset()

# set recursion limit so pickle doesn't error
sys.setrecursionlimit(40000)

random_state = np.random.RandomState(1999)
n_epochs = 200

# theano land tensor4 for 4 dimensions
input_var = tensor.tensor4('X')
target_var = tensor.tensor4('y')
outchan = y_train.shape[0]
inchan = X_train.shape[0]
width = X_train.shape[1]
height = X_train.shape[2]

input_var.tag.test_value = X_mb
target_var.tag.test_value = y_mb

# setting up theano - use None to indicate that dimension may change
coarse_input = InputLayer((minibatch_size, inchan, width, height),
                          input_var=input_var)
# choose number of filters and filter size
coarse_conv1 = Conv2DLayer(coarse_input, num_filters=32, filter_size=(5, 5),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(2,2))

coarse_pool1 = MaxPool2DLayer(coarse_conv1, pool_size=(2, 2))

coarse_conv2 = Conv2DLayer(coarse_pool1, num_filters=64, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(1,1))

coarse_pool2 = MaxPool2DLayer(coarse_conv2, pool_size=(2, 2))

coarse_conv3 = Conv2DLayer(coarse_pool2, num_filters=128, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(1,1))

coarse_conv4 = Conv2DLayer(coarse_conv3, num_filters=128, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(1,1))

coarse_conv5 = Conv2DLayer(coarse_conv4, num_filters=512, filter_size=(1, 1),
                      nonlinearity=rectify, W=GlorotUniform())

coarse_depool1 = Unpool2DLayer(coarse_conv5, (4,4))

coarse_deconv1 = TransposeConv2DLayer(coarse_depool1,
                                      num_filters=outchan,
                                      filter_size=(5,5),
                                      pad=(0,0),
                                      W=GlorotUniform(), nonlinearity=linear)

l_out = coarse_deconv1
#l_out = get_output(l_out)[:,:,:width,:height]
#theano.printing.Print("prediction SHAPE")(l_out.shape)
theano.printing.Print("coarse_pool1 SHAPE")(get_output(coarse_pool1).shape)
theano.printing.Print("coarse_pool2 SHAPE")(get_output(coarse_pool2).shape)
theano.printing.Print("coarse_depool1 SHAPE")(get_output(coarse_depool1).shape)
prediction = get_output(l_out)[:, :, :width, :height]
train_loss = squared_error(prediction, target_var)
train_loss = train_loss.mean()

valid_prediction = get_output(l_out, deterministic=True)[:, :, :width, :height]
valid_loss = squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

params = get_all_params(l_out, trainable=True)
# adam is the optimizer that is updating everything
updates = adam(train_loss, params, learning_rate=1E-4)

train_function = theano.function([input_var, target_var], train_loss,
                                 updates=updates)
valid_function = theano.function([input_var, target_var], valid_loss)
predict_function = theano.function([input_var], prediction)

checkpoint_dict = {}
checkpoint_dict["train_function"] = train_function
checkpoint_dict["valid_function"] = valid_function
checkpoint_dict["predict_function"] = predict_function


def _loop(function, itr):
    X_train, y_train = itr.next()
    ret = function(X_train, y_train)
    return [ret]


run_loop(_loop, train_function, train_itr, valid_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=100)
