# Kyle Kastner, Alec Radford
# License: MIT
"""
Convolutional VAE in a single file.
Bringing in code from IndicoDataSolutions and Alec Radford (NewMu)
Additionally converted to use default conv2d interface instead of explicit cuDNN
"""
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import conv2d
import tarfile
from time import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave, imread
import os

from skimage.transform import resize


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


def maxout(x):
    return T.maximum(x[:, 0::2], x[:, 1::2])


def clipped_maxout(x):
    return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -1., 1.)


def clipped_rectify(x):
    return T.clip((x + abs(x)) / 2.0, 0., 1.)


def hard_tanh(x):
    return T.clip(x, -1., 1.)


def steeper_sigmoid(x):
    return 1./(1. + T.exp(-3.75 * x))


def hard_sigmoid(x):
    return T.clip(x + 0.5, 0., 1.)


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


def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size/3)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1, 3))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def center_crop(img, n_pixels):
    img = img[n_pixels:img.shape[0] - n_pixels,
              n_pixels:img.shape[1] - n_pixels]
    return img


# wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
def lfw(n_imgs=1000, flatten=True, npx=64, datasets_dir='/Tmp/kastner'):
    data_dir = os.path.join(datasets_dir, 'lfw-deepfunneled')
    if (not os.path.exists(data_dir)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
        print('Downloading data from %s' % url)
        data_file = os.path.join(datasets_dir, 'lfw-deepfunneled.tgz')
        urllib.urlretrieve(url, data_file)
        tar = tarfile.open(data_file)
        os.chdir(datasets_dir)
        tar.extractall()
        tar.close()

    if n_imgs == 'all':
        n_imgs = 13233
    n = 0
    imgs = []
    Y = []
    n_to_i = {}
    for root, subFolders, files in os.walk(data_dir):
        if subFolders == []:
            if len(files) >= 2:
                for f in files:
                    if n < n_imgs:
                        if n % 1000 == 0:
                            print n
                        path = os.path.join(root, f)
                        img = imread(path) / 255.
                        img = resize(center_crop(img, 50), (npx, npx, 3)) - 0.5
                        if flatten:
                            img = img.flatten()
                        imgs.append(img)
                        n += 1
                        name = root.split('/')[-1]
                        if name not in n_to_i:
                            n_to_i[name] = len(n_to_i)
                        Y.append(n_to_i[name])
                    else:
                        break
    imgs = np.asarray(imgs, dtype=theano.config.floatX)
    imgs = imgs.transpose(0, 3, 1, 2)
    Y = np.asarray(Y)
    i_to_n = dict(zip(n_to_i.values(), n_to_i.keys()))
    return imgs, Y, n_to_i, i_to_n


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


def Adam(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
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

srng = RandomStreams()

trX, _, _, _ = lfw(n_imgs='all', flatten=False, npx=64)

trX = floatX(trX)


def log_prior(mu, log_sigma):
    """
    yaost kl divergence penalty
    """
    return 0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))


def conv(X, w, b, activation):
    #z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)


def conv_and_pool(X, w, b=None, activation=rectify):
    return max_pool_2d(conv(X, w, b, activation=activation), (2, 2))


def deconv(X, w, b=None):
    #z = dnn_conv(X, w, direction_hint="*not* 'forward!",
    #             border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z


def depool(X, factor=2):
    """
    luke perforated upsample
    http://www.brml.org/uploads/tx_sibibtex/281.pdf
    """
    output_shape = [
        X.shape[1],
        X.shape[2]*factor,
        X.shape[3]*factor
    ]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows*factor + (rows/stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                 output_shape[1], output_shape[2]))

    return upsamp


def deconv_and_depool(X, w, b=None, activation=rectify):
    return activation(deconv(depool(X), w, b))

n_code = 512
n_hidden = 2048
n_batch = 128

print('generating weights')

we = uniform((64, 3, 5, 5))
w2e = uniform((128, 64, 5, 5))
w3e = uniform((256, 128, 5, 5))
w4e = uniform((256 * 8 * 8, n_hidden))
b4e = shared0s(n_hidden)
wmu = uniform((n_hidden, n_code))
bmu = shared0s(n_code)
wsigma = uniform((n_hidden, n_code))
bsigma = shared0s(n_code)

wd = uniform((n_code, n_hidden))
bd = shared0s((n_hidden))
w2d = uniform((n_hidden, 256 * 8 * 8))
b2d = shared0s((256 * 8 * 8))
w3d = uniform((128, 256, 5, 5))
w4d = uniform((64, 128, 5, 5))
wo = uniform((3, 64, 5, 5))

enc_params = [we, w2e, w3e, w4e, b4e, wmu, bmu, wsigma, bsigma]
dec_params = [wd, bd, w2d, b2d, w3d, w4d, wo]
params = enc_params + dec_params


def conv_gaussian_enc(X, w, w2, w3, w4, b4, wmu, bmu, wsigma, bsigma):
    h = conv_and_pool(X, w)
    h2 = conv_and_pool(h, w2)
    h3 = conv_and_pool(h2, w3)
    h3 = h3.reshape((h3.shape[0], -1))
    h4 = T.tanh(T.dot(h3, w4) + b4)
    mu = T.dot(h4, wmu) + bmu
    log_sigma = 0.5 * (T.dot(h4, wsigma) + bsigma)
    return mu, log_sigma


def deconv_dec(X, w, b, w2, b2, w3, w4, wo):
    h = rectify(T.dot(X, w) + b)
    h2 = rectify(T.dot(h, w2) + b2)
    h2 = h2.reshape((h2.shape[0], 256, 8, 8))
    h3 = deconv_and_depool(h2, w3)
    h4 = deconv_and_depool(h3, w4)
    y = deconv_and_depool(h4, wo, activation=hard_tanh)
    return y


def model(X, e):
    code_mu, code_log_sigma = conv_gaussian_enc(X, *enc_params)
    Z = code_mu + T.exp(code_log_sigma) * e
    y = deconv_dec(Z, *dec_params)
    return code_mu, code_log_sigma, Z, y

print('theano code')

X = T.tensor4()
e = T.matrix()
Z_in = T.matrix()

code_mu, code_log_sigma, Z, y = model(X, e)

y_out = deconv_dec(Z_in, *dec_params)

rec_cost = T.sum(T.abs_(X - y))
prior_cost = log_prior(code_mu, code_log_sigma)

cost = rec_cost - prior_cost

print('getting updates')

updates = Adam(params, cost)

print('compiling')

_train = theano.function([X, e], cost, updates=updates)
_reconstruct = theano.function([X, e], y)
_x_given_z = theano.function([Z_in], y_out)
_z_given_x = theano.function([X, e], Z)

xs = floatX(np.random.randn(100, n_code))

print('TRAINING')

x_rec = floatX(shuffle(trX)[:100])

t = time()
n = 0.
n_epochs = 1000
for e in range(n_epochs):
    costs = []
    for xmb in iter_data(trX, size=n_batch):
        xmb = floatX(xmb)
        cost = _train(xmb, floatX(np.random.randn(xmb.shape[0], n_code)))
        costs.append(cost)
        n += xmb.shape[0]
    print(e, np.mean(costs), n / (time() - t))

    def tf(x):
        return ((x + 1.) / 2.).transpose(1, 2, 0)

    if e == n_epochs or e % 100 == 0:
        samples_path = os.path.join(os.path.split(__file__)[0],
                                    "sample_images_epoch_%d" % e)
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        samples = _x_given_z(xs)
        recs = _reconstruct(x_rec, floatX(np.ones((x_rec.shape[0], n_code))))
        img1 = color_grid_vis(x_rec,
                              transform=tf, show=False)
        img2 = color_grid_vis(recs,
                              transform=tf, show=False)
        img3 = color_grid_vis(samples,
                              transform=tf, show=False)

        imsave(os.path.join(samples_path, 'source.png'), img1)
        imsave(os.path.join(samples_path, 'recs.png'), img2)
        imsave(os.path.join(samples_path, 'samples.png'), img3)

        paths = make_paths(n_code, 9)
        for i in range(paths.shape[1]):
            path_samples = _x_given_z(floatX(paths[:, i, :]))
            for j, sample in enumerate(path_samples):
                imsave(os.path.join(
                    samples_path,  'paths_%d_%d.png' % (i, j)),
                    tf(sample))
