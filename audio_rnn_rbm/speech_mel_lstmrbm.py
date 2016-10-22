# Author: Kratarth Goel
# BITS Pilani (2014)
# LSTM-RBM for music generation

import sys
import os
import tables
import tarfile
import fnmatch
import random
import numpy
import numpy as np
from scipy.io import wavfile
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from midify import melspec, invmelspec, soundsc, stft, istft
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


def load_fruitspeech():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'audio.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "audio")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()

    h5_file_path = os.path.join(data_path, "saved_fruit.h5")
    if not os.path.exists(h5_file_path):
        data_path = os.path.join(data_path, "audio")

        audio_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.wav'):
                audio_matches.append(os.path.join(root, filename))

        random.seed(1999)
        random.shuffle(audio_matches)

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for wav_path in audio_matches:
            # Convert chars to int classes
            word = wav_path.split(os.sep)[-1][:-6]
            chars = [ord(c) - 97 for c in word]
            data_y.append(np.array(chars, dtype='int32'))
            fs, d = wavfile.read(wav_path)
            d = d.astype('float64') / (2 ** 15)
            # Preprocessing from A. Graves "Towards End-to-End Speech
            # Recognition"
            data_x.append(d.astype('float32'))
        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_y = h5_file.root.data_y

    # FIXME: HACKING
    train_x = data_x
    train_y = data_y
    valid_x = data_x
    valid_y = data_y
    test_x = data_x
    test_y = data_y
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    mask = rng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
    return x * mask

def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

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
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot( fast_dropout( rng, v) , W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot( fast_dropout(rng, h) , W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_lstmrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''Construct a symbolic RNN-RBM and initialize parameters.

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
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM (used during training)
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
  generation function.'''

    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    Wui = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqi = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wci = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bi = shared_zeros(n_hidden_recurrent)
    Wuf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wcf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bf = shared_zeros(n_hidden_recurrent)
    Wuc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bc = shared_zeros(n_hidden_recurrent)
    Wuo = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqo = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wco = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wqh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    bo = shared_zeros(n_hidden_recurrent)


    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, Wui, Wqi, Wci, bi,
    Wuf, Wqf, Wcf, bf, Wuc, Wqc, bc, Wuo, Wqo, Wco, bo , Wqv, Wqh
    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units
    q0 = T.zeros((n_hidden_recurrent,))
    c0 = T.zeros((n_hidden_recurrent,))


    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1, q_tm1, c_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv) + T.dot( q_tm1, Wqv)
        bh_t = bh + T.dot(u_tm1, Wuh) + T.dot( q_tm1, Wqh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))

        i_t = T.tanh(bi + T.dot(c_tm1, Wci) + T.dot(q_tm1, Wqi) + T.dot(u_t, Wui))
        f_t = T.tanh(bf + T.dot(c_tm1, Wcf) + T.dot(q_tm1, Wqf) + T.dot(u_t, Wuf))
        c_t = (f_t * c_tm1) + ( i_t * T.tanh( T.dot(u_t, Wuc) + T.dot( q_tm1, Wqc) + bc ))
        o_t = T.tanh(bo + T.dot(c_t, Wco) + T.dot(q_tm1, Wqo) + T.dot(u_t, Wuo))
        q_t = o_t * T.tanh( c_t )

        return ([v_t, u_t, q_t, c_t], updates) if generate else [u_t, q_t, c_t, bv_t, bh_t]

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
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
sequences.'''

    def __init__(self, n_vis, n_hidden=150, n_hidden_recurrent=100, lr=0.001):
        '''Constructs and compiles Theano functions for training and sequence
generation.

n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.
lr : float
  Learning rate
r : (integer, integer) tuple
  Specifies the pitch range of the piano-roll in MIDI note numbers, including
  r[0] but not r[1], such that r[1]-r[0] is the number of visible units of the
  RBM at a given time step. The default (21, 109) corresponds to the full range
  of piano (88 notes).
dt : float
  Sampling period when converting the MIDI files into piano-rolls, or
  equivalently the time difference between consecutive time steps.'''

        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_lstmrbm(n_vis, n_hidden,
                                           n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params,
                                                                gradient)))
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)

    def train(self, dataset, batch_size=100, num_epochs=200):
        print len(dataset)
        print len(dataset[0])
        print (dataset[0])
        try:
            for epoch in xrange(num_epochs):
                numpy.random.shuffle(dataset)
                costs = []

                for s, sequence in enumerate(dataset):
                    for i in xrange(0, len(sequence), batch_size):
                        cost = self.train_function(sequence[i:i + batch_size])
                        costs.append(cost)

                print 'Epoch %i/%i' % (epoch + 1, num_epochs),
                print numpy.mean(costs)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'

    def generate(self):
        generated = self.generate_function()
        return generated


def test_lstmrbm(X, batch_size=100, num_epochs=200):
    model = LstmRbm(X[0].shape[1])
    model.train(X, batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    train, valid, test = load_fruitspeech()
    train_x, train_y = train
    # load into main memory
    train_x = train_x[:]
    fs = 8000
    n_mel = 64
    n_fft = 256
    real_mel_samples = []
    imag_mel_samples = []
    for n, d in enumerate(train_x):
        Pxx = stft(d)
        rPxx = Pxx.real
        iPxx = Pxx.imag
        real_mel_spectrogram = melspec(rPxx, fs, n_mel)
        imag_mel_spectrogram = melspec(iPxx, fs, n_mel)
        slice_sz = 10
        lo = slice_sz // 2
        hi = len(real_mel_spectrogram) - slice_sz // 2
        if lo > hi:
            # ??? Shouldn't happen unless sequence is too short
            continue
        r = np.random.randint(lo, hi)
        real_mel_samples.append(
            real_mel_spectrogram[r - slice_sz // 2:r + slice_sz // 2])
        imag_mel_samples.append(
            imag_mel_spectrogram[r - slice_sz // 2:r + slice_sz // 2])

    #real_mel_samples = np.asarray(real_mel_samples).flatten()[:, None]
    #imag_mel_samples = np.asarray(imag_mel_samples).flatten()[:, None]
    real_mel_samples = np.asarray(real_mel_samples)
    real_mel_samples = real_mel_samples.reshape(
        real_mel_samples.shape[0] * real_mel_samples.shape[1], -1)
    imag_mel_samples = np.asarray(imag_mel_samples)
    imag_mel_samples = imag_mel_samples.reshape(
        imag_mel_samples.shape[0] * imag_mel_samples.shape[1], -1)

    print("Fitting Kmeans...")
    n_clusters = 40
    real_tf = KMeans(n_clusters=n_clusters, random_state=1999)
    real_tf.fit(real_mel_samples)
    imag_tf = KMeans(n_clusters=n_clusters, random_state=1999)
    imag_tf.fit(imag_mel_samples)

    print("Generating dataset")
    X = []
    for n, d in enumerate(train_x):
        Pxx = stft(d)
        rPxx = Pxx.real
        iPxx = Pxx.imag
        real_mel_spectrogram = melspec(rPxx, fs, n_mel)
        imag_mel_spectrogram = melspec(iPxx, fs, n_mel)
        length, n_features = real_mel_spectrogram.shape
        #real_mel_labels = real_tf.predict(
        #    real_mel_spectrogram.flatten()[:, None])
        #imag_mel_labels = imag_tf.predict(
        #    imag_mel_spectrogram.flatten()[:, None])
        #real_mel_labels = real_mel_labels.reshape((length, n_features))
        #imag_mel_labels = imag_mel_labels.reshape((length, n_features))
        real_mel_labels = real_tf.predict(
            real_mel_spectrogram)
        imag_mel_labels = imag_tf.predict(
            imag_mel_spectrogram)
        real_m_one_hot = np.zeros((length, n_clusters)).astype(
            theano.config.floatX)
        imag_m_one_hot = np.zeros((length, n_clusters)).astype(
            theano.config.floatX)
        for n in np.arange(length):
            real_m_one_hot[n, real_mel_labels[n]] = 1.
            imag_m_one_hot[n, imag_mel_labels[n]] = 1.
        m_one_hot = np.concatenate((real_m_one_hot, imag_m_one_hot), axis=1)
        X.append(m_one_hot)

    def reconstruct(codebook, one_hot, n_features):
        arr = np.zeros((len(one_hot), n_features))
        idx = np.where(one_hot)[1]
        for i in range(len(arr)):
            arr[i] = codebook[idx[i]]
        return arr

    real_mel = reconstruct(real_tf.cluster_centers_, X[0][:, :n_clusters],
                           n_features)
    imag_mel = reconstruct(imag_tf.cluster_centers_, X[0][:, -n_clusters:],
                           n_features)
    rPxx = invmelspec(real_mel, fs, n_fft)
    iPxx = invmelspec(imag_mel, fs, n_fft)
    Pxx = rPxx + 1j * iPxx
    Xs = istft(Pxx).astype('float32')
    lb = np.percentile(Xs, 0.5)
    ub = np.percentile(Xs, 99.5)
    Xs[Xs < lb] = 0.
    Xs[Xs > ub] = 0.
    Xs -= Xs.mean()
    wavfile.write('orig.wav', fs, soundsc(Xs))

    model = test_lstmrbm(X, num_epochs=200)
    n_samples_to_gen = 3
    for i in range(n_samples_to_gen):
        g = model.generate()
        real_mel = reconstruct(real_tf.cluster_centers_, g[:, :n_clusters],
                               n_features)
        imag_mel = reconstruct(imag_tf.cluster_centers_, g[:, -n_clusters:],
                               n_features)
        rPxx = invmelspec(real_mel, fs)
        iPxx = invmelspec(imag_mel, fs)
        Pxx = rPxx + 1j * iPxx
        Xs = istft(Pxx).astype('float32')
        lb = np.percentile(Xs, 0.5)
        ub = np.percentile(Xs, 99.5)
        Xs[Xs < lb] = 0.
        Xs[Xs > ub] = 0.
        Xs -= Xs.mean()
        wavfile.write('sample_%i.wav' % i, fs, soundsc(Xs))
