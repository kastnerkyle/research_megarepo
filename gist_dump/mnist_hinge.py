#!/usr/bin/env python
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import MonitorBased
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.datasets import mnist
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
from pylearn2.space import VectorSpace
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split


trn = mnist.MNIST(which_set = 'train',
                  one_hot = True,
                  axes = ['b', 0, 1, 'c'])

tst = mnist.MNIST(which_set = 'test',
                  one_hot = True,
                  axes = ['b', 0, 1, 'c'])

def is_a_zero(y, one_hot = True):
    targets = np.argmax(y, axis=1)
    out = np.zeros((y.shape[0], 2), dtype='float32')
    for n, t in enumerate(targets):
        def test(t):
            if t in [0, 1, 2, 3, 4]:
                return True
            else:
                return False
        out[n, test(t)] = 1.
    return out

trn = DenseDesignMatrix(topo_view=trn.get_topological_view(trn.X), y=is_a_zero(trn.y), axes=('b', 0, 1, 'c'))
tst = DenseDesignMatrix(topo_view=tst.get_topological_view(tst.X), y=is_a_zero(tst.y), axes=('b', 0, 1, 'c'))

l1 = mlp.RectifiedLinear(layer_name='l1',
                irange=.005,
                dim=512)

l2 = mlp.RectifiedLinear(layer_name='l2',
                irange=.005,
                dim=512)

#output = mlp.Softmax(n_classes=2,
#                     layer_name='y',
#                     irange=.005)

output = mlp.HingeLoss(layer_name='y',
                       irange=.005)

layers = [l1, l2, output]

mdl = mlp.MLP(layers,
              nvis=trn.X.shape[1])

lr = .01
epochs = 50
trainer = sgd.SGD(learning_rate=lr,
                  batch_size=200,
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=MonitorBased(
                      channel_name='valid_y_misclass',
                      prop_decrease=0.,
                      N=epochs),
                  monitoring_dataset={'valid': tst,
                                      'train': trn})

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='saved_clf.pkl')

experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher])

experiment.main_loop()