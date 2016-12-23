from scipy.linalg import svd
from scipy.io import wavfile
from scipy.cluster.vq import vq
from scipy.signal import firwin
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import tarfile
from numpy.lib.stride_tricks import as_strided
import theano
import os
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def fetch_sample_music():
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype(theano.config.floatX) / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    return fs, d


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype(theano.config.floatX) / (2 ** 15)
        speech.append(d)
    return fs, speech


def fetch_sample_speech_eustace(n_samples=None):
    """
    http://www.cstr.ed.ac.uk/projects/eustace/download.html
    """
    # data
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_wav.zip"
    wav_path = "eustace_wav.zip"
    if not os.path.exists(wav_path):
        download(url, wav_path)

    # labels
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_labels.zip"
    labels_path = "eustace_labels.zip"
    if not os.path.exists(labels_path):
        download(url, labels_path)

    # Read wavfiles
    # 16 kHz wav
    zf = zipfile.ZipFile(wav_path, 'r')
    wav_names = [fname for fname in zf.namelist()
                 if ".wav" in fname.split(os.sep)[-1]]
    fs = 16000
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        wav_str = zf.read(wav_name)
        d = np.frombuffer(wav_str, dtype=np.int16)
        d = d.astype(theano.config.floatX) / (2 ** 15)
        speech.append(d)

    zf = zipfile.ZipFile(labels_path, 'r')
    label_names = [fname for fname in zf.namelist()
                   if ".lab" in fname.split(os.sep)[-1]]
    labels = []
    print("Loading label files...")
    for label_name in label_names[:n_samples]:
        label_file_str = zf.read(label_name)
        labels.append(label_file_str)
    return fs, speech


def stft(X, fftsize=128, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()
    X = halfoverlap(X, fftsize)
    X = X * np.hanning(X.shape[-1])[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, mean_normalize=True, real=False,
          compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = np.fft.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = np.fft.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2] = X
        X_pad[:, fftsize // 2:] = 0
        X = X_pad
    X = local_ifft(X)
    X = invert_halfoverlap(X)
    if mean_normalize:
        X -= np.mean(X)
    return X


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def csvd(arr):
    """
    Do the complex SVD of a 2D array, returning real valued U, S, VT

    http://stemblab.github.io/complex-svd/
    """
    C_r = arr.real
    C_i = arr.imag
    block_x = C_r.shape[0]
    block_y = C_r.shape[1]
    K = np.zeros((2 * block_x, 2 * block_y))
    # Upper left
    K[:block_x, :block_y] = C_r
    # Lower left
    K[:block_x, block_y:] = C_i
    # Upper right
    K[block_x:, :block_y] = -C_i
    # Lower right
    K[block_x:, block_y:] = C_r
    return svd(K, full_matrices=False)


def icsvd(U, S, VT):
    """
    Invert back to complex values from the output of csvd

    U, S, VT = csvd(X)
    X_rec = inv_csvd(U, S, VT)
    """
    K = U.dot(np.diag(S)).dot(VT)
    block_x = U.shape[0] // 2
    block_y = U.shape[1] // 2
    arr_rec = np.zeros((block_x, block_y)) + 0j
    arr_rec.real = K[:block_x, :block_y]
    arr_rec.imag = K[:block_x, block_y:]
    return arr_rec


def plot_it(arr, scale=None, title="", cmap="gray"):
    if scale is "specgram":
        # plotting part
        mag = 10. * np.log10(np.abs(arr))
        # Transpose so time is X axis, and invert y axis so
        # frequency is low at bottom
        mag = mag.T[::-1, :]
    else:
        mag = arr
    f, ax = plt.subplots()
    ax.matshow(mag, cmap=cmap)
    plt.axis("off")
    x1 = mag.shape[0]
    y1 = mag.shape[1]

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)


def soundsc(X, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-1, 1) scaled version of X as float32, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    return X.astype('float32')


def herz_to_mel(freqs):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (freqs < bark_freq)
    mel = 0. * freqs
    mel[lin_pts] = (freqs[lin_pts] - f_0) / f_sp
    mel[~lin_pts] = bark_pt + np.log(freqs[~lin_pts] / bark_freq) / np.log(
        log_step)
    return mel


def mel_to_herz(mel):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(mel, np.ndarray):
        mel = np.array(mel)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (mel < bark_pt)

    freqs = 0. * mel
    freqs[lin_pts] = f_0 + f_sp * mel[lin_pts]
    freqs[~lin_pts] = bark_freq * np.exp(np.log(log_step) * (
        mel[~lin_pts] - bark_pt))
    return freqs


def mel_freq_weights(n_fft, fs, n_filts=None, width=None):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    min_freq = 0
    max_freq = fs // 2
    if width is None:
        width = 1.
    if n_filts is None:
        n_filts = int(herz_to_mel(max_freq) / 2) + 1
    else:
        n_filts = int(n_filts)
        assert n_filts > 0
    weights = np.zeros((n_filts, n_fft))
    fft_freqs = np.arange(n_fft // 2) / n_fft * fs
    min_mel = herz_to_mel(min_freq)
    max_mel = herz_to_mel(max_freq)
    partial = np.arange(n_filts + 2) / (n_filts + 1) * (max_mel - min_mel)
    bin_freqs = mel_to_herz(min_mel + partial)
    bin_bin = np.round(bin_freqs / fs * (n_fft - 1))
    for i in range(n_filts):
        fs_i = bin_freqs[i + np.arange(3)]
        fs_i = fs_i[1] + width * (fs_i - fs_i[1])
        lo_slope = (fft_freqs - fs_i[0]) / float(fs_i[1] - fs_i[0])
        hi_slope = (fs_i[2] - fft_freqs) / float(fs_i[2] - fs_i[1])
        weights[i, :n_fft // 2] = np.maximum(
            0, np.minimum(lo_slope, hi_slope))
    # Constant amplitude multiplier
    weights = np.diag(2. / (bin_freqs[2:n_filts + 2]
                      - bin_freqs[:n_filts])).dot(weights)
    weights[:, n_fft // 2:] = 0
    return weights


def time_attack_agc(X, fs, t_scale=0.5, f_scale=1.):
    """
    AGC based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    # 32 ms grid for FFT
    n_fft = 2 ** int(np.log(0.032 * fs) / np.log(2))
    f_scale = float(f_scale)
    window_size = n_fft
    window_step = window_size // 2
    X_freq = stft(X, window_size, mean_normalize=False)
    fft_fs = fs / window_step
    n_bands = max(10, 20 / f_scale)
    mel_width = f_scale * n_bands / 10.
    f_to_a = mel_freq_weights(n_fft, fs, n_bands, mel_width)
    f_to_a = f_to_a[:, :n_fft // 2]
    audiogram = np.abs(X_freq).dot(f_to_a.T)
    fbg = np.zeros_like(audiogram)
    state = np.zeros((audiogram.shape[1],))
    alpha = np.exp(-(1. / fft_fs) / t_scale)
    for i in range(len(audiogram)):
        state = np.maximum(alpha * state, audiogram[i])
        fbg[i] = state

    sf_to_a = np.sum(f_to_a, axis=0)
    E = np.diag(1. / (sf_to_a + (sf_to_a == 0)))
    E = E.dot(f_to_a.T)
    E = fbg.dot(E.T)
    E[E <= 0] = np.min(E[E > 0])
    ts = istft(X_freq / E, window_size, mean_normalize=False)
    return ts, X_freq, E


def hebbian_kmeans(X, n_clusters=10, n_epochs=10, W=None, learning_rate=0.01,
                   batch_size=100, random_state=None, verbose=True):
    """
    Modified from existing code from R. Memisevic
    See http://www.cs.toronto.edu/~rfm/code/hebbian_kmeans.py
    """
    if W is None:
        if random_state is None:
            random_state = np.random.RandomState()
        W = 0.1 * random_state.randn(n_clusters, X.shape[1])
    else:
        assert n_clusters == W.shape[0]
    X2 = (X ** 2).sum(axis=1, keepdims=True)
    last_print = 0
    for e in range(n_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_i = X[i: i + batch_size]
            X2_i = X2[i: i + batch_size]
            D = -2 * np.dot(W, X_i.T)
            D += (W ** 2).sum(axis=1, keepdims=True)
            D += X2_i.T
            S = (D == D.min(axis=0)[None, :]).astype("float").T
            W += learning_rate * (
                np.dot(S.T, X_i) - S.sum(axis=0)[:, None] * W)
        if verbose:
            if e == 0 or e > (.05 * n_epochs + last_print):
                last_print = e
                print("Epoch %i of %i, cost %.4f" % (
                    e + 1, n_epochs, D.min(axis=0).sum()))
    return W


def complex_to_real_view(arr_c):
    # Inplace view from complex to r, i as separate columns
    assert arr_c.dtype in [np.complex64, np.complex128]
    shp = arr_c.shape
    dtype = np.float64 if arr_c.dtype == np.complex128 else np.float32
    arr_r = arr_c.ravel().view(dtype=dtype).reshape(shp[0], 2 * shp[1])
    return arr_r


def real_to_complex_view(arr_r):
    # Inplace view from real, image as columns to complex
    assert arr_r.dtype not in [np.complex64, np.complex128]
    shp = arr_r.shape
    dtype = np.complex128 if arr_r.dtype == np.float64 else np.complex64
    arr_c = arr_r.ravel().view(dtype=dtype).reshape(shp[0], shp[1] // 2)
    return arr_c


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def polyphase_core(x, m, f):
    # x = input data
    # m = decimation rate
    # f = filter
    # Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    polyphase = p = np.zeros((m, (x.shape[0] + f.shape[0]) / m), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    # Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p


def polyphase_single_filter(x, m, f):
    return np.sum(polyphase_core(x, m, f), axis=0)


def polyphase_lowpass(arr, downsample=2, n_taps=50, filter_pad=1.1):
    filt = firwin(downsample * n_taps, 1 / (downsample * filter_pad))
    filtered = polyphase_single_filter(arr, downsample, filt)
    return filtered


def window(arr, window_size, window_step=1, axis=0):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>

    <http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter>
    """
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > arr.shape[-1]:
        raise ValueError("`window_size` is too long.")

    orig = list(range(len(arr.shape)))
    trans = list(range(len(arr.shape)))
    trans[axis] = orig[-1]
    trans[-1] = orig[axis]
    arr = arr.transpose(trans)

    shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    strides = arr.strides + (arr.strides[-1],)
    strided = as_strided(arr, shape=shape, strides=strides)

    if window_step > 1:
        strided = strided[..., ::window_step, :]

    orig = list(range(len(strided.shape)))
    trans = list(range(len(strided.shape)))
    trans[-2] = orig[-1]
    trans[-1] = orig[-2]
    trans = trans[::-1]
    strided = strided.transpose(trans)
    return strided


def unwindow(arr, window_size, window_step=1, axis=0):
    # undo windows by broadcast
    if axis != 0:
        raise ValueError("axis != 0 currently unsupported")
    shp = arr.shape
    unwindowed = np.tile(arr[:, None, ...], (1, window_step, 1, 1))
    unwindowed = unwindowed.reshape(shp[0] * window_step, *shp[1:])
    return unwindowed.mean(axis=1)


if __name__ == "__main__":
    def _pre(list_of_data):
        n_fft = 256
        f_c = np.vstack([stft(dd, n_fft) for dd in list_of_data])
        f_r = complex_to_real_view(f_c)
        return f_r, n_fft

    def preprocess_train(list_of_data, random_state):
        f_r, n_fft = _pre(list_of_data)
        clusters = f_r
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_fft = _pre(list_of_data)
        memberships, distances = vq(f_r, clusters)
        vq_r = clusters[memberships]
        vq_c = real_to_complex_view(vq_r)
        d_k = istft(vq_c, fftsize=n_fft)
        return d_k

    random_state = np.random.RandomState(1999)

    """
    # Doesn't work yet for unknown reasons...
    fs, d = fetch_sample_music()
    sub = int(.8 * d.shape[0])
    d1 = [d[:sub]]
    d2 = [d[sub:]]
    """

    fs, d = fetch_sample_speech_fruit()
    d1 = d[::8] + d[1::8] + d[2::8] + d[3::8] + d[4::8] + d[5::8] + d[6::8]
    d2 = d[7::8]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]

    clusters = preprocess_train(d1, random_state)
    # Training data
    vq_d1 = apply_preprocess(d1, clusters)
    vq_d2 = apply_preprocess(d2, clusters)
    assert [i != j for i, j in zip(vq_d2.ravel(), vq_d2.ravel())]

    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)

    wavfile.write("train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("test_no_agc.wav", fs, soundsc(fix_d2))
    wavfile.write("vq_test_no_agc.wav", fs, soundsc(vq_d2, fs))

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    wavfile.write("train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("vq_test_agc.wav", fs, soundsc(agc_vq_d2))
