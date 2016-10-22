# Author: Kratarth Goel
# BITS Pilani (2014)
# LSTM-RBM for music generation

import glob
import os
import sys
import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.shared_randomstreams import RandomStreams
from kdllib import gradient_clipping, make_weights, make_biases
from kdllib import sgd, adadelta, adam
from kdllib import fetch_fruitspeech_spectrogram
from kdllib import midiwrap, fetch_nottingham
import cPickle as pickle

midiread, midiwrite = midiwrap()

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

def build_rbm(v, W, bv, bh, k):
    '''
    Construct a k-step Gibbs chain starting at v for an RBM.

    v : Theano vector or matrix
      If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
      Weight matrix of the RBM.
    bv : Theano vector
      Visible bias vector of the RBM.
    bh : Theano vector
      Hidden bias vector of the RBM.
    k : scalar or Theano scalar
      Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample : Theano vector or matrix with the same shape as `v`
      Corresponds to the generated sample(s).
    cost : Theano scalar
      Expression whose gradient with respect to W, bv, bh is the CD-k
      approximation to the log-likelihood of `v` (training example)
      under the RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
      Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
      The `updates` object returned by scan.
    '''

    def gibbs_step(v, debug=False):
        mean_h = tensor.nnet.sigmoid(v.dot(W) + bh)
        if debug:
            h = tensor.zeros_like(mean_h)
        else:
            h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                             dtype=theano.config.floatX)
        mean_v = tensor.nnet.sigmoid(h.dot(W.T) + bv)
        if debug:
            v = tensor.zeros_like(mean_v)
        else:
            v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                             dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = tensor.xlogx.xlogy0(v, mean_v) + tensor.xlogx.xlogy0(
                1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - tensor.log(
                 1 + tensor.exp(v.dot(W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]
    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''
    Initialize a matrix shared variable with normally distributed
    elements.
    '''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''
    Initialize a vector shared variable with zero elements.
    '''
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))


def build_lstmrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''
    Construct a symbolic RNN-RBM and initialize parameters.

    n_visible : integer
      Number of visible units.
    n_hidden : integer
      Number of hidden units of the conditional RBMs.
    n_hidden_recurrent : integer
      Number of hidden units of the RNN.

    Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
              updates_generate) tuple:

    v : Theano matrix
      Symbolic variable holding an input sequence (used during training)
    v_sample : Theano matrix
      Symbolic variable holding the negative particles for CD log-likelihood
      gradient estimation (used during training)
    cost : Theano scalar
      Expression whose gradient (considering v_sample constant) corresponds to
      the LL gradient of the RNN-RBM (used during training)
    monitor : Theano scalar
      Frame-level pseudo-likelihood (useful for monitoring during training)
    params : tuple of Theano shared variables
      The parameters of the model to be optimized during training.
    updates_train : dictionary of Theano variable -> Theano variable
      Update object that should be passed to theano.function when compiling the
      training function.
    v_t : Theano matrix
      Symbolic variable holding a generated sequence (used during sampling)
    updates_generate : dictionary of Theano variable -> Theano variable
      Update object that should be passed to theano.function when compiling the
      generation function.
    '''
    random_state = np.random.RandomState(1999)
    W, = make_weights(n_visible, [n_hidden], random_state, init="normal",
                      scale=0.01)
    bv, bh = make_biases([n_visible, n_hidden])

    scale = 0.0001
    Wuh, Wuv = make_weights(n_hidden_recurrent, [n_hidden, n_visible],
                            random_state, init="normal", scale=scale)
    Wvu, = make_weights(n_visible, [n_hidden_recurrent,], random_state,
                        init="normal", scale=scale)

    Wuu, Wui, Wqi, Wci, Wuf, Wqf, Wcf, Wuc, Wqc, Wuo, Wqo, Wco = make_weights(
        n_hidden_recurrent, [n_hidden_recurrent] * 12, random_state,
        init="normal", scale=scale)
    Wqv, Wqh = make_weights(n_hidden_recurrent, [n_visible, n_hidden],
                            random_state, init="normal", scale=scale)
    bu, bi, bf, bc, bo = make_biases([n_hidden_recurrent] * 5)

    params = [W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, Wui, Wqi, Wci, bi,
              Wuf, Wqf, Wcf, bf, Wuc, Wqc, bc, Wuo, Wqo, Wco, bo , Wqv, Wqh]
    # learned parameters as shared
    # variables

    v = tensor.matrix()  # a training sequence
    u0 = tensor.zeros((n_hidden_recurrent,))  # initial value for the RNN
                                              # hidden units
    q0 = tensor.zeros((n_hidden_recurrent,))
    c0 = tensor.zeros((n_hidden_recurrent,))


    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1, q_tm1, c_tm1):
        bv_t = bv + u_tm1.dot(Wuv) + q_tm1.dot(Wqv)
        bh_t = bh + u_tm1.dot(Wuh) + q_tm1.dot(Wqh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(tensor.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = tensor.tanh(bu + v_t.dot(Wvu) + u_tm1.dot(Wuu))

        i_t = tensor.tanh(bi + c_tm1.dot(Wci) + q_tm1.dot(Wqi) + u_t.dot(Wui))
        f_t = tensor.tanh(bf + c_tm1.dot(Wcf) + q_tm1.dot(Wqf) + u_t.dot(Wuf))
        c_t = (f_t * c_tm1) + (i_t * tensor.tanh(u_t.dot(Wuc) + q_tm1.dot(Wqc) + bc))
        o_t = tensor.tanh(bo + c_t.dot(Wco) + q_tm1.dot(Wqo) + u_t.dot(Wuo))
        q_t = o_t * tensor.tanh(c_t)
        if generate:
            return ([v_t, u_t, q_t, c_t], updates)
        else:
            return [u_t, q_t, c_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.

    (u_t, q_t, c_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, q_tm1, c_tm1, *_: recurrence(v_t, u_tm1, q_tm1, c_tm1),
        sequences=v, outputs_info=[u0, q0, c0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t, q_t, c_t), updates_generate = theano.scan(
        lambda u_tm1, q_tm1, c_tm1, *_: recurrence(None, u_tm1, q_tm1, c_tm1),
        outputs_info=[None, u0, q0, c0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class LstmRbm:
    '''
    Simple class to train an RNN-RBM to generate sample
    sequences.
    '''
    def __init__(self, n_visible=88, n_hidden=150, n_hidden_recurrent=100,
                 lr=0.0001):
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_lstmrbm(n_visible, n_hidden,
                                           n_hidden_recurrent)

        grads = tensor.grad(cost, params, consider_constant=[v_sample])
        """
        opt = sgd(params, lr)
        """

        """
        opt = adadelta(params)
        grads = gradient_clipping(grads, 10.)
        """

        opt = adam(params, lr)
        grads = gradient_clipping(grads, 10.)
        updates = opt.updates(params, grads)
        updates_train.update(updates)
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)

    def train(self, data, minibatch_size=100, n_epochs=200):
        '''
        Data should be a list of 2D numpy arrays
        '''
        try:
            for epoch in range(n_epochs):
                np.random.shuffle(data)
                costs = []

                for s, sequence in enumerate(data):
                    for i in range(0, len(sequence), minibatch_size):
                        cost = self.train_function(
                            sequence[i:i + minibatch_size].astype(
                            theano.config.floatX))
                        costs.append(cost)

                print('Epoch %i/%i' % (epoch + 1, n_epochs))
                print(np.mean(costs))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print('Interrupted by user.')

    def generate(self):
        gen = self.generate_function()
        return gen


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    minibatch_size = 100
    n_epochs = 200
    random_state = np.random.RandomState(1999)

    key_range, dt, dataset = fetch_nottingham()
    if not os.path.exists('saved.pkl'):
        model = LstmRbm(n_visible=dataset[0].shape[-1])
        model.train(dataset, minibatch_size=minibatch_size, n_epochs=n_epochs)
        cur = sys.getrecursionlimit()
        sys.setrecursionlimit(40000)
        with open('saved.pkl', mode='w') as f:
            pickle.dump(model, f, -1)
        sys.setrecursionlimit(cur)

    with open('saved.pkl', mode='r') as f:
        model = pickle.load(f)
    piano_roll = model.generate()
    midiwrite("generation1.midi", piano_roll, key_range, dt)
