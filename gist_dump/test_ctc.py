"""
bitmap utils and much of the ctc code modified from Shawn Tan
"""
# Author: Kyle Kastner
# License: BSD 3-clause
from theano import tensor
from scipy import linalg
import theano
import numpy as np
import matplotlib.pyplot as plt

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

chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
mapping = {c: i for i, c in enumerate(chars)}


def string_to_image(string):
    return np.hstack(np.array([bitmap[mapping[c]] for c in string])).T[:, ::-1]


def string_to_index(string):
    return [mapping[c] for c in string]


def logplus_(log_a, log_b):
    # Returns log (a + b)
    return log_a + tensor.log(1 + tensor.exp(log_b - log_a))


def log_(a):
    return tensor.log(tensor.clip(a, eps, 1))


def exp_(a):
    return tensor.exp(tensor.clip(a, np.log(eps), 30))


def log_path_probs(y_hat, y):
    eye = tensor.eye(y.shape[0])
    first = eye[0]
    mask0 = 1 - eye[0]
    mask1 = 1 - eye[1]
    alt_mask = tensor.cast(tensor.arange(y.shape[0]) % 2, theano.config.floatX)
    skip_mask = mask0 * mask1 * alt_mask
    prev_idx = tensor.arange(-1, y.shape[0] - 1)
    prev_prev_idx = tensor.arange(-2, y.shape[0] - 2)
    log_mask0 = log_(mask0)
    log_skip_mask = log_(skip_mask)
    log_first = log_(first)

    def step(log_p_curr, log_p_prev):
        log_after_trans = logplus_(log_p_prev, logplus_(
                    log_mask0 + log_p_prev[prev_idx],
                    log_skip_mask + log_p_prev[prev_prev_idx])
        )
        log_p_next = log_p_curr + log_after_trans
        return log_p_next

    L = tensor.log(y_hat[:, y])
    log_f_probs, _ = theano.scan(step, sequences=[L], outputs_info=[log_first])
    log_b_probs, _ = theano.scan(step, sequences=[L[::-1, ::-1]],
                                 outputs_info=[log_first])

    log_probs = log_f_probs + log_b_probs[::-1, ::-1]
    return log_probs, prev_idx, prev_prev_idx


def log_ctc_cost(y_hat, y):
    log_probs, prev_idx, prev_prev_idx = log_path_probs(y_hat, y)
    max_log_prob = tensor.max(log_probs)

    norm_probs = tensor.exp(log_probs - max_log_prob)
    norm_total_log_prob = tensor.log(tensor.sum(norm_probs))

    log_total_prob = norm_total_log_prob + max_log_prob
    return -log_total_prob


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


def build_tanh_rnn(hidden_inputs, W_hidden_hidden, b_hidden, initial_hidden):
    def step(input_curr, hidden_prev):
        hidden = tensor.tanh(tensor.dot(hidden_prev, W_hidden_hidden) +
                             input_curr + b_hidden)
        return hidden
    hidden, _ = theano.scan(step,
                            sequences=[hidden_inputs],
                            outputs_info=[initial_hidden])
    return hidden


def build_model(X, input_size, hidden_size, output_size):
    random_state = np.random.RandomState(1999)
    W_input_hidden = as_shared(np_tanh_fan((input_size, hidden_size),
                                           random_state))
    W_hidden_hidden = as_shared(np_ortho((hidden_size, hidden_size),
                                         random_state))
    W_hidden_output = as_shared(np_tanh_fan((hidden_size, output_size),
                                            random_state))
    b_hidden = as_shared(np_zeros((hidden_size,)))
    i_hidden = as_shared(np_zeros((hidden_size,)))
    b_output = as_shared(np_zeros((output_size,)))
    hidden = build_tanh_rnn(tensor.dot(X, W_input_hidden), W_hidden_hidden,
                            b_hidden, i_hidden)
    predict = tensor.nnet.softmax(tensor.dot(hidden, W_hidden_output)
                                  + b_output)
    params = [W_input_hidden, W_hidden_hidden, W_hidden_output, b_hidden,
              i_hidden, b_output]
    return X, predict, params


def label_seq(string):
    idxs = string_to_index(string)
    blank = -1
    result = np.ones((len(idxs) * 2 + 1,), dtype=np.int32) * blank
    result[np.arange(len(idxs)) * 2 + 1] = idxs
    return result


def theano_label_seq(y):
    y_ext = y.dimshuffle((0, 'x'))
    blank = -1
    blanks = tensor.zeros_like(y_ext) + blank
    concat = tensor.concatenate([blanks, blanks], axis=1).flatten()
    concat = tensor.concatenate([concat, blanks[0]], axis=0).flatten()
    indices = 2 * tensor.arange(y_ext.shape[0]) + 1
    concat = tensor.set_subtensor(concat[indices], y_ext.flatten())
    return concat


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


def ctc_prediction_to_string(y_pred):
    indices = y_pred.argmax(axis=1)
    # remove blanks
    indices = indices[indices != len(chars)]
    # remove repeats
    not_same = np.where((indices[1:] != indices[:-1]))[0]
    last_char = ""
    if len(not_same) > 0:
        last_char = chars[indices[-1]]
        indices = indices[not_same]
    s = "".join([chars[i] for i in indices])
    return s + last_char


def prediction_to_string(y_pred):
    indices = y_pred.argmax(axis=1)
    # remove blanks
    indices = indices[indices != len(chars)]
    s = "".join([chars[i] for i in indices])
    return s


if __name__ == "__main__":
    X_sym = tensor.matrix('X')
    y_sym = tensor.ivector('Y_s')
    X, predict, params = build_model(X_sym, 8, 256, len(chars) + 1)

    y_ctc = theano_label_seq(y_sym)
    cost = log_ctc_cost(predict, y_ctc)

    grads = tensor.grad(cost, wrt=params)
    opt = adadelta(params)
    train = theano.function(inputs=[X_sym, y_sym], outputs=cost,
                            updates=opt.updates(params, grads))
    pred = theano.function(inputs=[X_sym], outputs=predict)
    string = "Hello"
    X = string_to_image(string)
    y = string_to_index(string)
    for i in range(1000):
        print("Iteration %i:" % i)
        print(train(X, y))
        p = pred(X)
        print(prediction_to_string(p))
        print(ctc_prediction_to_string(p))
    print("Final prediction:")
    p = pred(X)
    print(prediction_to_string(p))
    print(ctc_prediction_to_string(p))
    plt.matshow(X.T[::-1], cmap="gray")
    plt.title(ctc_prediction_to_string(p) + " : " + string)
    plt.show()
