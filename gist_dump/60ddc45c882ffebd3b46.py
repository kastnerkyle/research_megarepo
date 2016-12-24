from scipy.linalg import svd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
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


def stft(X, fftsize=128, mean_normalize=True, compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if compute_onesided:
        local_fft = np.fft.rfft
        fftsize = 2 * fftsize
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if mean_normalize:
        X -= X.mean()
    X = halfoverlap(X, fftsize)
    X = X * np.hanning(X.shape[-1])[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, mean_normalize=True, compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if compute_onesided:
        local_ifft = np.fft.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
    else:
        local_ifft = np.fft.ifft
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


def plot_it(arr, title=""):
    # plotting part
    mag = 10. * np.log10(np.abs(arr))
    # Transpose so time is X axis, and invert y axis so frequency is low at bottom
    mag = mag.T[::-1, :]
    f, ax = plt.subplots()
    ax.matshow(mag, cmap="gray")
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


if __name__ == "__main__":
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    random_state = np.random.RandomState(1999)
    fs, d = wavfile.read(wav_path)
    d = d.astype(theano.config.floatX) / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    f_d = stft(d, fftsize=256)
    d2 = istft(f_d, fftsize=256)
    wavfile.write("orig.wav", fs, soundsc(d2))

    print("Calculating noise on STFT components")
    c_n = 1E-1

    def nrange(arr):
        return arr.std() #arr.max() - arr.min()

    corr_r_d = f_d
    corr_r_d += c_n * random_state.randn(*f_d.shape) * nrange(f_d.real)
    corr_r = istft(corr_r_d, fftsize=256)
    wavfile.write("corr_r.wav", fs, soundsc(corr_r))

    corr_i_d = f_d
    corr_i_d += c_n * 1j * random_state.randn(*f_d.shape) * nrange(f_d.imag)
    corr_i = istft(corr_i_d, fftsize=256)
    wavfile.write("corr_i.wav", fs, soundsc(corr_i))

    corr_b_d = f_d
    corr_b_d += c_n * 1j * random_state.randn(*f_d.shape) * nrange(np.abs(f_d))
    corr_b_d += c_n * random_state.randn(*f_d.shape) * nrange(np.abs(f_d))
    corr_b = istft(corr_b_d, fftsize=256)
    wavfile.write("corr_b.wav", fs, soundsc(corr_b))

    plot_it(f_d, "Original")
    plt.savefig("orig.png")
    plt.close()

    U, S, VT = csvd(f_d)
    for n in [1E-4, 1E-3, 1E-2, 1E-1, 0., 1., 10.]:
        for i in range(3):
            Ui = U
            Si = S
            VTi = VT
            if i == 0:
                coeff = n * nrange(Ui)
                Ui += coeff * random_state.randn(*Ui.shape)
                noise_name = "U"
            elif i == 1:
                coeff = n * nrange(Si)
                Si += coeff * random_state.randn(*Si.shape)
                noise_name = "S"
            elif i == 2:
                coeff = n * nrange(VTi)
                VTi += coeff * random_state.randn(*VTi.shape)
                noise_name = "VT"
            f_d_rec = icsvd(Ui, Si, VTi)
            d_rec = istft(f_d_rec, fftsize=256)

            substr = "%f * mean of %s noise on %s" % (n, noise_name, noise_name)
            print("Calculating " + substr)
            plot_it(f_d_rec, "Reconstructed, " + substr)
            noisestr = "%.1e" % n
            wavfile.write("rec_%s_%s.wav" % (noisestr, noise_name), fs, soundsc(d_rec))
            plt.savefig("rec_%s_%s.png" % (noisestr, noise_name))
            plt.close()
