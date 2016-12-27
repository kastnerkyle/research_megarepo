"""
bitmap utils from Shawn Tan
"""
# Author: Kyle Kastner
# License: BSD 3-clause
# Equation 46 from http://arxiv.org/abs/1308.0850
from theano import tensor
from scipy import linalg
import theano
import numpy as np
import matplotlib.pyplot as plt
import time

eps = 1E-12

characters = np.array([
    0x0,
    0x808080800080000,
    0x2828000000000000,
    0x287C287C280000,
    0x81E281C0A3C0800,
    0x6094681629060000,
    0x1C20201926190000,
    0x808000000000000,
    0x810202010080000,
    0x1008040408100000,
    0x2A1C3E1C2A000000,
    0x8083E08080000,
    0x81000,
    0x3C00000000,
    0x80000,
    0x204081020400000,
    0x1824424224180000,
    0x8180808081C0000,
    0x3C420418207E0000,
    0x3C420418423C0000,
    0x81828487C080000,
    0x7E407C02423C0000,
    0x3C407C42423C0000,
    0x7E04081020400000,
    0x3C423C42423C0000,
    0x3C42423E023C0000,
    0x80000080000,
    0x80000081000,
    0x6186018060000,
    0x7E007E000000,
    0x60180618600000,
    0x3844041800100000,
    0x3C449C945C201C,
    0x1818243C42420000,
    0x7844784444780000,
    0x3844808044380000,
    0x7844444444780000,
    0x7C407840407C0000,
    0x7C40784040400000,
    0x3844809C44380000,
    0x42427E4242420000,
    0x3E080808083E0000,
    0x1C04040444380000,
    0x4448507048440000,
    0x40404040407E0000,
    0x4163554941410000,
    0x4262524A46420000,
    0x1C222222221C0000,
    0x7844784040400000,
    0x1C222222221C0200,
    0x7844785048440000,
    0x1C22100C221C0000,
    0x7F08080808080000,
    0x42424242423C0000,
    0x8142422424180000,
    0x4141495563410000,
    0x4224181824420000,
    0x4122140808080000,
    0x7E040810207E0000,
    0x3820202020380000,
    0x4020100804020000,
    0x3808080808380000,
    0x1028000000000000,
    0x7E0000,
    0x1008000000000000,
    0x3C023E463A0000,
    0x40407C42625C0000,
    0x1C20201C0000,
    0x2023E42463A0000,
    0x3C427E403C0000,
    0x18103810100000,
    0x344C44340438,
    0x2020382424240000,
    0x800080808080000,
    0x800180808080870,
    0x20202428302C0000,
    0x1010101010180000,
    0x665A42420000,
    0x2E3222220000,
    0x3C42423C0000,
    0x5C62427C4040,
    0x3A46423E0202,
    0x2C3220200000,
    0x1C201804380000,
    0x103C1010180000,
    0x2222261A0000,
    0x424224180000,
    0x81815A660000,
    0x422418660000,
    0x422214081060,
    0x3C08103C0000,
    0x1C103030101C0000,
    0x808080808080800,
    0x38080C0C08380000,
    0x324C000000,
], dtype=np.uint64)

bitmap = np.unpackbits(characters.view(np.uint8)).reshape(characters.shape[0],
                                                          8, 8)
bitmap = bitmap[:, ::-1, :]

chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`"
chars += "abcdefghijklmnopqrstuvwxyz{|}~"
mapping = {c: i for i, c in enumerate(chars)}


def string_to_image(string):
    return np.hstack(np.array([bitmap[mapping[c]] for c in string])).T[:, ::-1]


def string_to_index(string):
    return np.asarray([mapping[c] for c in string])


def index_to_onehot(indexes):
    o = np.zeros(
        (indexes.shape[0], indexes.shape[1], len(chars))).astype("int32")
    for n, i in enumerate(indexes):
        o[n, np.arange(indexes.shape[1]), i] = 1
    return o


def onehot_to_index(one_hot):
    return np.argmax(one_hot, axis=-1)


class adadelta(object):
    """
    An adaptive learning rate optimizer

    For more information, see:
    Matthew D. Zeiler, "ADADELTA: An Adaptive Learning Rate Method"
    arXiv:1212.5701.
    """
    def __init__(self, params, running_grad_decay=0.95, running_up_decay=0.95,
                 eps=1E-6):
        self.running_grad_decay = running_grad_decay
        self.running_up_decay = running_up_decay
        self.eps = eps
        self.running_up2_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.running_grads2_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.previous_grads_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]

    def updates(self, params, grads):
        running_grad_decay = self.running_grad_decay
        running_up_decay = self.running_up_decay
        eps = self.eps
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            running_grad2 = self.running_grads2_[n]
            running_up2 = self.running_up2_[n]
            previous_grad = self.previous_grads_[n]
            rg2up = running_grad_decay * running_grad2 + (
                1. - running_grad_decay) * (grad ** 2)
            updir = -tensor.sqrt(running_up2 + eps) / tensor.sqrt(
                running_grad2 + eps) * previous_grad
            ru2up = running_up_decay * running_up2 + (
                1. - running_up_decay) * (updir ** 2)
            updates.append((previous_grad, grad))
            updates.append((running_grad2, rg2up))
            updates.append((running_up2, ru2up))
            updates.append((param, param + updir))
        return updates


def make_minibatch_from_strings(strings):
    X_shapes = [string_to_image(s).shape for s in strings]
    y_shapes = [string_to_index(s).shape for s in strings]
    max_X_len = max([sh[0] for sh in X_shapes])
    max_y_len = max([sh[0] for sh in y_shapes])
    minibatch_size = len(strings)
    # assume all feature dimensions are equal!
    X_mb = np.zeros((max_X_len, minibatch_size, X_shapes[-1][1])).astype(
        theano.config.floatX)
    X_mask = np.zeros((max_X_len, len(strings))).astype(theano.config.floatX)
    y_mb = np.zeros((max_y_len, minibatch_size)).astype("int32")
    y_mask = np.ones_like(y_mb).astype(theano.config.floatX)
    for n, s in enumerate(strings):
        X = string_to_image(s)
        y = string_to_index(s)
        X_mb[:X.shape[0], n, :] = X
        X_mask[:X.shape[0], n] = 1.
        y_mb[:y.shape[0], n] = y
        y_mask[:y.shape[0], n] = 1.
    return X_mb, X_mask, y_mb, y_mask


def as_shared(arr, name=None):
    if type(arr) in [float, int]:
        if name is not None:
            return theano.shared(np.cast[theano.config.floatX](arr))
        else:
            return theano.shared(np.cast[theano.config.floatX](arr), name=name)
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def np_zeros(shape):
    """ Builds a numpy variable filled with zeros """
    return np.zeros(shape).astype(theano.config.floatX)


def np_ones(shape):
    """ Builds a numpy variable filled with zeros """
    return np.ones(shape).astype(theano.config.floatX)


def np_rand(shape, random_state):
    # Make sure bounds aren't the same
    return random_state.uniform(low=-0.08, high=0.08, size=shape).astype(
        theano.config.floatX)


def np_randn(shape, random_state):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * random_state.randn(*shape)).astype(theano.config.floatX)


def np_tanh_fan(shape, random_state):
    # The . after the 6 is critical! shape has dtype int...
    bound = np.sqrt(6. / np.sum(shape))
    return random_state.uniform(low=-bound, high=bound,
                                size=shape).astype(theano.config.floatX)


def np_sigmoid_fan(shape, random_state):
    return 4 * np_tanh_fan(shape, random_state)


def np_ortho(shape, random_state):
    """ Builds a theano variable filled with orthonormal random values """
    g = random_state.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def build_model(X, X_mask, y, y_mask, minibatch_size, input_size,
                cond_size, hidden_size, output_size, n_gaussians=5):
    random_state = np.random.RandomState(1999)
    # Input to hidden weights
    W_input_hidden = as_shared(np_tanh_fan((input_size, hidden_size),
                                           random_state))
    b_input = as_shared(np_zeros((hidden_size,)))

    # Conditioning to hidden weights
    W_cond_hidden = as_shared(np_tanh_fan((cond_size, hidden_size),
                                          random_state))
    b_cond = as_shared(np_zeros((hidden_size,)))

    # alpha, beta, kappa = 3 x n_gaussians
    W_window = as_shared(np_tanh_fan((hidden_size, 3 * n_gaussians),
                                     random_state))
    b_window = as_shared(np_zeros((3 * n_gaussians),))

    # Hidden to hidden
    W_hidden_hidden = as_shared(np_ortho((hidden_size, hidden_size),
                                         random_state))

    # Hidden to output
    W_hidden_output = as_shared(np_tanh_fan((hidden_size, output_size),
                                            random_state))
    b_output = as_shared(np_zeros((output_size,)))

    # Zeros init
    initial_hidden = as_shared(np_zeros((minibatch_size, hidden_size)))
    initial_kappa = as_shared(np_zeros((minibatch_size, n_gaussians)))

    proj_X = tensor.dot(X, W_input_hidden) + b_input

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        # need to be *very* careful
        dim = n_gaussians
        if arr.ndim == 3:
            return arr[:, :, n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    def step(x_t, xm_t, h_tm1, k_tm1, c, cm, U, W_c, b_c, W_ch, b_ch, s):
        window_ti = tensor.dot(h_tm1, W_c) + b_c
        alpha_t = tensor.exp(_slice(window_ti, 0))
        beta_t = tensor.exp(_slice(window_ti, 1))
        # No log 1 + gives very different results...
        kappa_t = k_tm1 + tensor.exp(
            _slice(window_ti, 2) + 1E-6)
        theano.printing.Print("k")(kappa_t.shape)
        theano.printing.Print("s")(s.shape)
        sq_tx = (kappa_t[:, None, :] - s[None, :, None]) ** 2
        theano.printing.Print("sq_tx")(sq_tx.shape)
        theano.printing.Print("alpha_t")(alpha_t.shape)
        theano.printing.Print("beta_t")(beta_t.shape)
        mixture_t = alpha_t[:, None, :] * tensor.exp(
            -beta_t[:, None, :] * sq_tx)
        theano.printing.Print("mixture_t")(mixture_t.shape)
        phi_tx = mixture_t.sum(axis=2)
        theano.printing.Print("phi_tx")(phi_tx.shape)
        theano.printing.Print("c")(c.shape)
        theano.printing.Print("cm")(cm.shape)
        masked_c = (c * cm[:, :, None])
        masked_c = masked_c.dimshuffle(1, 0, 2)
        theano.printing.Print("m_c")(masked_c.shape)
        theano.printing.Print("m_n")(phi_tx[:, :, None].shape)
        window_t = (phi_tx[:, :, None] * masked_c).sum(axis=1)
        theano.printing.Print("window_t")(window_t.shape)
        proj_w = tensor.dot(window_t, W_ch) + b_ch
        h_ti = tensor.tanh(x_t + tensor.dot(h_tm1, U) + proj_w)

        # Masking
        h_t = xm_t[:, None] * h_ti + (1. - xm_t[:, None]) * h_tm1
        return h_t, kappa_t, alpha_t, beta_t, phi_tx, mixture_t, window_t

    """
    rvals1 = step(proj_X[0], X_mask[0],
                  initial_hidden, initial_kappa,
                  y, y_mask,
                  W_hidden_hidden, W_window, b_window, W_cond_hidden, b_cond)
    h1, k1, a1, b1, p1, w1 = rvals1
    rvals2 = step(proj_X[1], X_mask[1],
                  h1, k1,
                  y, y_mask,
                  W_hidden_hidden, W_window, b_window, W_cond_hidden, b_cond)
    h2, k2, a2, b2, p2, w2 = rvals2
    params = [W_input_hidden, b_input, W_cond_hidden, b_cond,
              W_hidden_hidden, W_window, b_window, W_hidden_output, b_output]
    return X, h2, k2, a2, b2, p2, w2, params
    """
    steps = tensor.arange(y.shape[0], dtype=theano.config.floatX)
    theano.printing.Print("steps")(steps.shape)
    rvals, updates = theano.scan(step,
                                 sequences=[proj_X[:-1], X_mask[:-1]],
                                 outputs_info=[initial_hidden, initial_kappa,
                                               None, None, None, None, None],
                                 non_sequences=[y, y_mask,
                                                W_hidden_hidden,
                                                W_window, b_window,
                                                W_cond_hidden, b_cond,
                                                steps])
    h, k, a, b, p, m, w = rvals
    hidden_proj = tensor.dot(h, W_hidden_output) + b_output
    theano.printing.Print("hp")(hidden_proj.shape)
    params = [W_input_hidden, b_input, W_cond_hidden, b_cond,
              W_hidden_hidden, W_window, b_window, W_hidden_output, b_output]
    return X, hidden_proj, k, a, b, p, m, w, params


if __name__ == "__main__":
    base_string = "cat"
    import itertools
    true_strings = sorted(list(set(["".join(i) for i in [
        s for s in itertools.permutations(base_string)]])))
    print(true_strings)
    dataset_size = len(true_strings)
    minibatch_size = 2
    if dataset_size % int(minibatch_size) != 0:
        raise ValueError(
            "Minibatch size (%s) not an even multiple " % minibatch_size +
            "of dataset size (%s). Handle it!" % dataset_size)
    m = np.arange(0, dataset_size, minibatch_size)
    minibatch_indices = list(zip(m[:-1], m[1:])) + [(m[-1], dataset_size)]
    X, X_mask, y, y_mask = make_minibatch_from_strings(true_strings)

    y = index_to_onehot(y)
    y = y.astype(theano.config.floatX)
    X = X.astype(theano.config.floatX)

    X_sym = tensor.tensor3('X')
    X_mask_sym = tensor.matrix('X_mask')
    y_sym = tensor.tensor3('y')
    y_mask_sym = tensor.matrix('y_mask')

    i, j = minibatch_indices[0]
    X_sym.tag.test_value = X[:, i:j]
    X_mask_sym.tag.test_value = X_mask[:, i:j]
    y_sym.tag.test_value = y[:, i:j]
    y_mask_sym.tag.test_value = y_mask[:, i:j]

    X_res, hidden_proj, k, a, b, p, m, w, params = build_model(
        X_sym, X_mask_sym, y_sym, y_mask_sym,
        minibatch_size,
        X.shape[-1], y.shape[-1],
        256, X.shape[-1], n_gaussians=2)
    predict = tensor.nnet.sigmoid(hidden_proj)
    true = X_sym[1:]
    cost = (-true * tensor.log(predict) - (1 - true) * tensor.log(
        1. - predict)).sum(axis=2)
    cost = cost.sum(axis=0).mean()

    grads = tensor.grad(cost, wrt=params)
    opt = adadelta(params)
    train = theano.function(inputs=[X_sym, X_mask_sym, y_sym, y_mask_sym],
                            outputs=cost,
                            updates=opt.updates(params, grads))
    pred = theano.function(inputs=[X_sym, X_mask_sym, y_sym, y_mask_sym],
                           outputs=[predict, k, a, b, p, m, w])

    def plot_items(cost, ret, title, base_offset):
        pr, kappa, alpha, beta, phi, mixture, window = ret
        for x in range(minibatch_size):
            ts = time.time()
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].matshow(pr[:, x, ::-1].T, cmap="gray")
            axarr[1].matshow(alpha[:, x, :].T)
            for k in range(kappa.shape[-1]):
                x_offset = np.arange(len(kappa)) - 0.5
                axarr[2].plot(x_offset, np.log(kappa[:, x, k]),
                              label="N(%i)" % k)
            axarr[2].legend(loc=4)
            if cost is not None:
                plt.suptitle("%s, cost %f" % (title, cost))
            #f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]],
                      visible=False)
            plt.savefig("%s_%s_%s.png" % (base_offset + x, title, ts))
            plt.close()

    n_epochs = 2000
    for e in range(n_epochs):
        costs = []
        for n, (i, j) in enumerate(minibatch_indices):
            train_cost = train(X[:, i:j], X_mask[:, i:j],
                               y[:, i:j], y_mask[:, i:j])
            costs.append(train_cost)
            if e % 250 == 0 or e == (n_epochs - 1):
                print("Iteration %i, mb %i:" % (e, n))
                mean_cost = np.mean(costs)
                print(mean_cost)
                ret = pred(X[:, i:j], X_mask[:, i:j], y[:, i:j], y_mask[:, i:j])
                pr, kappa, alpha, beta, phi, mixture, window = ret
                plot_items(mean_cost, ret, "forced", i)
    print("Final cost")
    mean_cost = np.mean(costs)
    print(mean_cost)

    for n, (i, j) in enumerate(minibatch_indices):
        pred_X = np.zeros_like(X[:, i:j])
        start_slice = 2
        pred_X[:start_slice] = X[:start_slice, i:j]
        pred_X[:start_slice] = X[:start_slice, i:j]
        sub_y = np.zeros_like(y[:, i:j])
        sub_y[:] = y[:, i:j]
        sub_y_mask = np.zeros_like(y_mask[:, i:j])
        sub_y_mask[:] = y_mask[:, i:j]
        # use all ones mask since we have no idea in generation how long it is
        sub_X_mask = np.ones_like(X_mask[:, i:j])
        for ii in np.arange(start_slice, len(X) - 1):
            ret = pred(pred_X[:ii], sub_X_mask[:ii],
                       sub_y, sub_y_mask)
            pr, kappa, alpha, beta, phi, mixture, window = ret
            pr[pr < 0.5] = 0
            pr[pr >= 0.5] = 1
            pred_X[ii + 1, :] = pr[-1]
        ret = pred(pred_X, sub_X_mask, sub_y, sub_y_mask)
        plot_items(mean_cost, ret, "self", i)
    from IPython import embed
    embed()
    raise ValueError()
