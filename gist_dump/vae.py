# Alec Radford, Indico, Kyle Kastner
# License: MIT
"""
VAE in a single file.
Bringing in code from IndicoDataSolutions and Alec Radford (NewMu)
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tempfile
import gzip
import cPickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
from time import time
import os


def softmax(x):
    return T.nnet.softmax(x)


def rectify(x):
    return (x + abs(x)) / 2.0


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def linear(x):
    return x


def t_rectify(x):
    return x * (x > 1)


def t_linear(x):
    return x * (abs(x) > 1)


def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))


def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape) * scale)


def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])


def bw_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    sqrt_shp = int(np.sqrt(X.shape[1]))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        x = x.reshape((sqrt_shp, sqrt_shp))
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d


def mnist(datasets_dir='/Tmp/kastner'):
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def make_paths(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps/48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths


def Adam(params, cost, lr=0.0001, b1=0.1, b2=0.1, e=1e-8):
    """
    no bias init correction
    """
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    return updates


class PickleMixin(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


def log_prior(mu, log_sigma):
    """
    yaost kl divergence penalty
    """
    return 0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))


class VAE(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="snapshot.pkl",
                 sizes=[256, 128], n_code=64, n_batch=20):
        self.srng = RandomStreams()
        self.sizes = sizes
        self.n_code = n_code
        self.n_batch = n_batch
        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            f.close()

    def _setup_functions(self, trX):
        enc_tuples = []
        dec_tuples = []
        prev_size = trX.shape[1]
        for s in self.sizes:
            enc_t = (prev_size, s)
            dec_t = (s, prev_size)
            enc_tuples.append(enc_t)
            dec_tuples.append(dec_t)
            prev_size = s
        print(enc_tuples)
        print(dec_tuples)

        if not hasattr(self, "params"):
            print('generating weights')
            enc_params = []
            for n in range(len(enc_tuples)):
                w_enc = uniform(enc_tuples[n])
                b_enc = shared0s((enc_tuples[n][1],))
                enc_params += [w_enc, b_enc]

            wmu = uniform((self.sizes[-1], self.n_code))
            bmu = shared0s((self.n_code,))
            wsigma = uniform((self.sizes[-1], self.n_code))
            bsigma = shared0s((self.n_code,))
            enc_params += [wmu, bmu, wsigma, bsigma]
            self.enc_params = enc_params

            dec_params = []
            # stop = -1 to include 0
            for n in range(len(dec_tuples) - 1, -1, -1):
                # Reverse order due to end reversal
                w_dec = uniform(dec_tuples[n])
                b_dec = shared0s((dec_tuples[n][1],))
                dec_params += [w_dec, b_dec]

            wd = uniform((self.n_code, self.sizes[-1]))
            bd = shared0s((self.sizes[-1],))
            dec_params = [wd, bd] + dec_params

            self.dec_params = dec_params
            self.params = self.enc_params + self.dec_params

        print('theano code')
        X = T.matrix()
        e = T.matrix()
        X.tag.test_value = trX[:self.n_batch]
        e.tag.test_value = floatX(np.random.randn(self.n_batch, self.n_code))

        Z_in = T.matrix()
        Z_in.tag.test_value = floatX(np.random.randn(self.n_batch, self.n_code))

        code_mu, code_log_sigma, Z, y = self._model(X, e)
        y_out = self._dec(Z_in, self.dec_params)

        # rec_cost = T.sum(T.abs_(X - y))  # / T.cast(X.shape[0], 'float32')
        rec_cost = T.sum(T.sqr(X - y))  # / T.cast(X.shape[0], 'float32')
        prior_cost = log_prior(code_mu, code_log_sigma)

        lr = 1E-3  # * trX.shape[0]

        cost = rec_cost - prior_cost

        print('getting updates')

        updates = Adam(self.params, cost, lr)

        print('compiling')
        self._fit_function = theano.function([X, e], cost, updates=updates)
        self._reconstruct = theano.function([X, e], y)
        self._x_given_z = theano.function([Z_in], y_out)
        self._z_given_x = theano.function([X], (code_mu, code_log_sigma))

    def _gaussian_enc(self, X, enc_params):
        [w, b, w2, b2, w3, b3, wmu, bmu, wsigma, bsigma] = enc_params
        h = rectify(T.dot(X, w) + b)
        h2 = rectify(T.dot(h, w2) + b2)
        h3 = T.tanh(T.dot(h2, w3) + b3)
        mu = T.dot(h3, wmu) + bmu
        log_sigma = 0.5 * (T.dot(h3, wsigma) + bsigma)
        return mu, log_sigma

    def _dec(self, X, dec_params):
        [w, b, w2, b2, w3, b3, w4, b4] = dec_params
        h = rectify(T.dot(X, w) + b)
        h2 = rectify(T.dot(h, w2) + b2)
        h3 = rectify(T.dot(h2, w3) + b3)
        y = T.dot(h3, w4) + b4
        return y

    def _model(self, X, e):
        code_mu, code_log_sigma = self._gaussian_enc(X, self.enc_params)
        Z = code_mu + (T.exp(code_log_sigma) + 1e-6) * e
        y = self._dec(Z, self.dec_params)
        return code_mu, code_log_sigma, Z, y

    def fit(self, trX):
        if not hasattr(self, "_fit_function"):
            self._setup_functions(trX)

        xs = floatX(np.random.randn(100, self.n_code))
        print('TRAINING')
        x_rec = floatX(shuffle(trX)[:100])
        t = time()
        n = 0.
        epochs = 1000
        for e in range(epochs):
            for xmb in iter_data(trX, size=self.n_batch):
                xmb = floatX(xmb)
                cost = self._fit_function(xmb, floatX(
                    np.random.randn(xmb.shape[0], self.n_code)))
                self.costs_.append(cost)
                n += xmb.shape[0]
            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))
            self.epoch_ += 1

            if e % 5 == 0 or e == (epochs - 1):
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == epochs or e % 100 == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root
                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)
                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

                samples = self._x_given_z(xs)
                recs = self._reconstruct(x_rec, floatX(
                    np.random.randn(x_rec.shape[0], self.n_code)))

                img1 = bw_grid_vis(x_rec, show=False)
                img2 = bw_grid_vis(recs, show=False)
                img3 = bw_grid_vis(samples, show=False)

                imsave(os.path.join(samples_path, 'source.png'), img1)
                imsave(os.path.join(samples_path, 'recs.png'), img2)
                imsave(os.path.join(samples_path, 'samples.png'), img3)

                paths = make_paths(self.n_code, 3)
                for i in range(paths.shape[1]):
                    path_samples = self._x_given_z(floatX(paths[:, i, :]))
                    sqrt_shp = int(np.sqrt(x_rec.shape[1]))
                    for j, sample in enumerate(path_samples):
                        imsave(os.path.join(samples_path,
                                            'paths_%d_%d.png' % (i, j)),
                               sample.squeeze().reshape((sqrt_shp, sqrt_shp)))

    def transform(self, x_rec):
                recs = self._reconstruct(x_rec, floatX(
                    np.ones((x_rec.shape[0], self.n_code))))
                return recs

    def encode(self, X, e=None):
        if e is None:
            e = np.ones((X.shape[0], self.n_code))
        return self._z_given_x(X, e)

    def decode(self, Z):
        return self._z_given_x(Z)

if __name__ == "__main__":
    tr, _, _, = mnist()
    trX, trY = tr
    tf = VAE(image_save_root="/Tmp/kastner",
             snapshot_file="/Tmp/kastner/mnist_snapshot.pkl",
             sizes=[256, 128])
    trX = floatX(trX)
    tf.fit(trX)
    recs = tf.transform(trX[:100])
