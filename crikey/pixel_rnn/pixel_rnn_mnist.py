import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import load_checkpoint, conv2d, conv2d_transpose
from kdllib import make_weights, make_biases
from kdllib import make_conv_weights
from kdllib import list_iterator
from kdllib import fetch_binarized_mnist
from kdllib import theano_one_hot
from kdllib import softmax, tanh, logsumexp, sigmoid
from kdllib import adam, gradient_clipping
from kdllib import binary_crossentropy, categorical_crossentropy
from kdllib import run_loop


if __name__ == "__main__":
    import argparse

    mnist = fetch_binarized_mnist()
    X = mnist["data"]
    train_indices = mnist["train_indices"]
    valid_indices = mnist["valid_indices"]
    X = np.array([x.astype(theano.config.floatX) for x in X])

    minibatch_size = 16
    n_epochs = 10000  # Used way at the bottom in the training loop!
    checkpoint_every_n = 10
    n_bins = 1
    random_state = np.random.RandomState(1999)

    # bit weird but for MNIST this will return 28, 1, 28
    train_itr = list_iterator([X], minibatch_size, axis=1,
                              stop_index=train_indices[-1] + 1, randomize=True,
                              make_mask=True)
    valid_itr = list_iterator([X], minibatch_size, axis=1,
                              start_index=valid_indices[0],
                              stop_index=valid_indices[-1] + 1,
                              randomize=True, make_mask=True)
    X_mb, X_mb_mask = next(train_itr)
    train_itr.reset()

    desc = "Speech generation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--sample',
                        help='Sample from a checkpoint file',
                        default=None,
                        required=False)
    def restricted_int(x):
        if x is None:
            # None makes it "auto" sample
            return x
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("%r not range [1, inf]" % (x,))
        return x
    parser.add_argument('-sl', '--sample_length',
                        help='Number of steps to sample, default is automatic',
                        type=restricted_int,
                        default=None,
                        required=False)
    parser.add_argument('-c', '--continue', dest="cont",
                        help='Continue training from another saved model',
                        default=None,
                        required=False)
    args = parser.parse_args()
    if args.sample is not None:
        raise ValueError("Not yet implemented")
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask


    params = []
    biases = []

    n_conv1 = 128
    k_conv1 = (1, 1)
    k_conv1_hid = (1, 3)

    conv1_w, = make_conv_weights(1, [n_conv1,], k_conv1, random_state)
    conv1_b, = make_biases([n_conv1,])
    params += [conv1_w, conv1_b]
    biases += [conv1_b]

    # Might become 3* for GRU or 4* for LSTM
    conv1_hid, = make_conv_weights(n_conv1, [n_conv1,], k_conv1_hid, random_state)
    params += [conv1_hid]

    pred_w, = make_weights(n_conv1, [n_bins,], init="fan",
                           random_state=random_state)
    pred_b, = make_biases([n_bins])
    params += [pred_w, pred_b]
    biases += [pred_b]

    theano.printing.Print("X_sym.shape")(X_sym.shape)
    # add channel dim
    im = X_sym.dimshuffle(1, 'x', 0, 2)
    target = im
    shp = im.shape
    # careful shift to avoid leakage
    conv1 = conv2d(im, conv1_w, conv1_b, border_mode=(0, k_conv1[1]))
    theano.printing.Print("conv1.shape")(conv1.shape)
    conv1 = conv1[:, :, :, :shp[3]]
    theano.printing.Print("conv1.shape")(conv1.shape)
    r_conv1 = conv1.dimshuffle(2, 1, 0, 3)
    theano.printing.Print("r_conv1.shape")(r_conv1.shape)
    shp = r_conv1.shape

    init_hidden = tensor.zeros((minibatch_size, n_conv1, 1, shp[3]),
                                dtype=theano.config.floatX)
    # weirdness in broadcast
    if minibatch_size == 1:
        init_hidden = tensor.unbroadcast(init_hidden, 0, 2)
    else:
        init_hidden = tensor.unbroadcast(init_hidden, 2)
    theano.printing.Print("init_hidden.shape")(init_hidden.shape)
    # recurrent function (using tanh activation function)
    def step(in_t, h_tm1):
        theano.printing.Print("in_t.shape")(in_t.shape)
        theano.printing.Print("h_tm1.shape")(h_tm1.shape)
        h_i = conv2d(h_tm1, conv1_hid, border_mode="half")
        theano.printing.Print("h_i.shape")(h_i.shape)
        in_i = in_t.dimshuffle(1, 0, 'x', 2)
        theano.printing.Print("in_i.shape")(in_i.shape)
        h_t = tanh(in_i + h_i)
        # need to add broadcast dims back to keep scan happy
        theano.printing.Print("h_t.shape")(h_t.shape)
        return h_t
    h, updates = theano.scan(fn=step,
                             sequences=[r_conv1],
                             outputs_info=[init_hidden])
    h = tensor.unbroadcast(h, 0, 1, 2, 3, 4)
    # remove spurious axis
    h = h[:, :, :, 0]
    theano.printing.Print("h.shape")(h.shape)
    # dimshuffle to b01c
    h = h.dimshuffle(1, 0, 3, 2)
    theano.printing.Print("h.shape")(h.shape)
    pred = sigmoid(h.dot(pred_w) + pred_b)
    theano.printing.Print("pred.shape")(pred.shape)
    target = target.dimshuffle(0, 2, 3, 1)
    theano.printing.Print("target.shape")(target.shape)
    cost = binary_crossentropy(pred, target)
    theano.printing.Print("cost.shape")(cost.shape)
    cost = cost.dimshuffle(1, 2, 0)
    theano.printing.Print("cost.shape")(cost.shape)
    shp = cost.shape
    cost = cost.reshape((shp[0] * shp[1], -1))
    theano.printing.Print("cost.shape")(cost.shape)
    cost = cost.sum(axis=0).mean()
    theano.printing.Print("cost.shape")(cost.shape)

    l2_penalty = 0
    for p in list(set(params) - set(biases)):
        l2_penalty += (p ** 2).sum()

    reg_cost = cost + 1E-3 * l2_penalty

    grads = tensor.grad(reg_cost, params)
    grads = gradient_clipping(grads, 10.)

    learning_rate = 1E-4

    opt = adam(params, learning_rate)
    updates = opt.updates(params, grads)

    if args.cont is not None:
        print("Continuing training from saved model")
        continue_path = args.cont
        if not os.path.exists(continue_path):
            raise ValueError("Continue model %s, path not "
                             "found" % continue_path)
        saved_checkpoint = load_checkpoint(continue_path)
        checkpoint_dict = saved_checkpoint
        train_function = checkpoint_dict["train_function"]
        cost_function = checkpoint_dict["cost_function"]
        predict_function = checkpoint_dict["predict_function"]
    else:
        train_function = theano.function([X_sym, X_mask_sym],
                                         [cost],
                                         updates=updates,
                                         on_unused_input='warn')
        cost_function = theano.function([X_sym, X_mask_sym],
                                        [cost],
                                        on_unused_input='warn')
        predict_function = theano.function([X_sym, X_mask_sym],
                                           [pred],
                                           on_unused_input='warn')
        print("Beginning training loop")
        checkpoint_dict = {}
        checkpoint_dict["train_function"] = train_function
        checkpoint_dict["cost_function"] = cost_function
        checkpoint_dict["predict_function"] = predict_function


    def _loop(function, itr):
        X_mb, X_mb_mask = next(itr)
        rval = function(X_mb, X_mb_mask)
        cost = rval[0]
        return [cost]

run_loop(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=checkpoint_every_n, skip_minimums=True,
         skip_intermediates=True, skip_most_recents=True)
