# License: BSD 3-clause
# Authors: Kyle Kastner
from __future__ import print_function
import numpy as np
import uuid
from numpy.lib.stride_tricks import as_strided
from scipy import linalg, fftpack
from scipy.cluster.vq import kmeans, vq
from scipy.io import wavfile
import scipy.signal as sg
import shutil
from collections import Iterable, defaultdict
import socket
import wave
import tarfile
import zipfile
import os
import glob
import re
import copy
from collections import Counter
import time
import sys
import inspect
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools
import time
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import theano
import theano.tensor as tensor
from theano.tensor import nnet
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
# Doesn't support binomial with n > 1??
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.scan_module.scan_utils import infer_shape
from theano.gof.fg import MissingInputError
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib
import locale
import logging

sys.setrecursionlimit(40000)

"""
init logging
"""
logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)
"""
end logging
"""

"""
begin decorators
"""
def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start
"""
end decorators
"""
"""
begin metautils
"""
def get_generic_name():
    # may need to make this a search for the first non-kdllib reference
    # make generic name from highest calling context
    prev_function_name = None
    for i in range(len(inspect.stack())):
        (frame, filename, line_number,
         function_name, lines, index) = inspect.stack()[i]
        #print(frame, filename, line_number, function_name, lines, index)
        # Use stack to get easier function name than parsing the code itself
        if i > 0:
            _, _, _, prev_function_name, _, _ = inspect.stack()[i - 1]
        else:
            prev_function_name = function_name
        script_name = filename.split(os.sep)[-1]
        lib_location = os.path.realpath(__file__)
        lib_name = lib_location.split(os.sep)[-1]
        # cover .py and .pyc
        if script_name != lib_name and script_name != lib_name[:-1]:
            name = script_name + "_" + prev_function_name
            #print(frame, filename, line_number, function_name, lines, index)
            return name
    raise ValueError("Issue in generic name getter")


# Many of these from Ishaan G.
def print_param_info(params):
    """Print information about the parameters in the given param set."""

    params = sorted(params, key=lambda p: p.name)
    values = [p.get_value(borrow=True) for p in params]
    shapes = [p.shape for p in values]
    logger.info("Params:")
    for param, value, shape in zip(params, values, shapes):
        logger.info("\t%s (%s)" % (param.name, ",".join([str(x) for x in shape])))

    total_param_count = 0
    for shape in shapes:
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_param_count += param_count
    logger.info("Total parameter count: %f M" % (total_param_count / float(1E6)))


def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l


_params = {}
def param(name=None, *args, **kwargs):
    """
    A wrapper for `theano.shared` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `theano.shared`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.
    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """

    had_ext = False
    if type(name) is tuple or type(name) is list:
        name = flatten(name)
        name_ext = name[1:]
        name = name[0]
        had_ext = True

    if name is None:
        name = get_generic_name()

    name = name + "_1"
    sub = name.split("_")
    # skip purely numbers
    ss = []
    for s in sub:
        try:
            int(s)
        except:
            ss.append(s)
    sub = ss

    def key_fn(k):
        tt = k.split("_")[:len(sub)]
        return tt
    matches = [k for k in _params.keys()
               if key_fn(k) == sub]
    if len(matches) > 0:
        num = len(matches) + 1
        name = "_".join(sub) + "_%i" % num

    # Ability to append info
    if had_ext:
        ns = name.split("_")
        name = "_".join(ns[:-1] + name_ext + [ns[-1]])

    kwargs['name'] = name
    param = as_shared(*args, **kwargs)
    param.param = True
    param.name = name
    _params[name] = param
    return _params[name]


def delete_params(name):
    to_delete = [p_name for p_name in _params if name in p_name]
    for p_name in to_delete:
        del _params[p_name]


def param_search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function. When optimizing a cost function, you
    can use this to get a list of all of the trainable params in the graph, like
    so:
    `lib.search(cost, lambda x: hasattr(x, "param"))`
    """
    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, tensor.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results
    return _search(node, critereon, set())


def save_params(path):
    raise ValueError("fix it, unify with save coroutine")
    param_vals = {}
    for name, param in _params.iteritems():
        param_vals[name] = param.get_value()

    with open(path, 'wb') as f:
        pickle.dump(param_vals, f)


def load_params(path):
    raise ValueError("fix it, unify")
    with open(path, 'rb') as f:
        param_vals = pickle.load(f)

    for name, val in param_vals.iteritems():
        _params[name].set_value(val)


def clear_all_params():
    to_delete = [p_name for p_name in _params]
    for p_name in to_delete:
        del _params[p_name]

"""
end metautils
"""

"""
begin datasets
"""
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
    X = .9 * X
    X = X * 2 ** 15
    return X.astype('int16')


def get_script_name():
    script_path = os.path.abspath(sys.argv[0])
    # Assume it ends with .py ...
    script_name = script_path.split(os.sep)[-1]
    return script_name


def get_resource_dir(name, resource_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not resource_dir:
        resource_dir = os.getenv("KDLLIB_DIR", os.path.join(
            os.path.expanduser("~"), "kdllib_dir"))
    if folder is None:
        resource_dir = os.path.join(resource_dir, name)
    else:
        resource_dir = os.path.join(resource_dir, folder)
    if create_dir:
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
    return resource_dir


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
            logger.info("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        logger.info("Downloading: %s Bytes: %s" % (server_fname, file_size))
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
                logger.info(status)
                p += progress_update_percentage


def midiwrap():
    """
    Wrapper to midi read and midi write
    """
    try:
        sys.path.insert(1, get_resource_dir(""))
        from midi.utils import midiread, midiwrite
        sys.path.pop(1)
    except ImportError:
        logger.info("Need GPL licensed midi utils, downloading...",
              "http://www.iro.umontreal.ca/~lisa/deep/midi.zip")
        url = "http://www.iro.umontreal.ca/~lisa/deep/midi.zip"
        partial_path = get_resource_dir("")
        full_path = os.path.join(partial_path, "midi.zip")
        if not os.path.exists(full_path):
            download(url, full_path)
            zip_ref = zipfile.ZipFile(full_path, 'r')
            zip_ref.extractall(partial_path)
            zip_ref.close()
        sys.path.insert(1, get_resource_dir(""))
        from midi.utils import midiread, midiwrite
        sys.path.pop(1)
    return midiread, midiwrite


class BlizzardThread(threading.Thread):
    """Blizzard Thread"""
    def __init__(self, queue, out_queue, preproc_fn):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue
        self.preproc_fn = preproc_fn

    def run(self):
        while True:
            # Grabs image path from queue
            wav_paths, texts = self.queue.get()
            text_group = texts
            wav_group = [wavfile.read(wp)[1] for wp in wav_paths]
            wav_group = [w.astype('float32') / (2 ** 15) for w in wav_group]
            wav_group = [self.preproc_fn(wi) for wi in wav_group]
            self.out_queue.put((wav_group, text_group))
            self.queue.task_done()


class Blizzard_dataset(object):
    def __init__(self, minibatch_size=2,
                 blizzard_path='/home/kkastner/blizzard_data'):
        self.n_fft = 256
        self.n_step = self.n_fft // 4
        self.blizzard_path = blizzard_path
        # extracted text
        self.text_path = os.path.join(self.blizzard_path, 'train', 'segmented',
                                      'prompts.gui')
        with open(self.text_path, 'r') as f:
            tt = f.readlines()
            wav_names = [t.strip() for t in tt[::3]]
            raw_other = tt[2::3]
            raw_text = [t.strip().lower() for t in tt[1::3]]
            all_symbols = set()
            for rt in raw_text:
                all_symbols = set(list(all_symbols) + list(set(rt)))
            self.wav_names = wav_names
            self.text = raw_text
            self.symbols = sorted(list(all_symbols))
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        raise ValueError()
        # These files come from converting the Blizzard mp3 files to wav,
        # then placing in a directory called blizzard_wav
        self.wav_paths = glob.glob(os.path.join(self.blizzard_path,
                                                'blizzard_wav', '*.wav'))
        self.minibatch_size = minibatch_size
        self._lens = np.array([float(len(t)) for t in self.text])

        # Get only the smallest 50% of files for now
        _cut = np.percentile(self._lens, 5)
        _ind = np.where(self._lens <= _cut)[0]

        self.text = [self.text[i] for i in _ind]
        self.wav_names = [self.wav_names[i] for i in _ind]
        assert len(self.text) == len(self.wav_names)
        final_wav_paths = []
        final_text = []
        final_wav_names = []
        for n, (w, t) in enumerate(zip(self.wav_names, self.text)):
            parts = w.split("chp")
            name = parts[0]
            chapter = [pp for pp in parts[1].split("_") if pp != ''][0]
            for p in self.wav_paths:
                if name in p and chapter in p:
                    final_wav_paths.append(p)
                    final_wav_names.append(w)
                    final_text.append(t)
                    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
                    raise ValueError()
                    break

        # resort into shortest -> longest order
        sorted_inds = np.argsort([len(t) for t in final_text])
        st = [final_text[i] for i in sorted_inds]
        swp = [final_wav_paths[i] for i in sorted_inds]
        swn = [final_wav_names[i] for i in sorted_inds]
        self.wav_names = swn
        self.wav_paths = swp
        self.text = st
        assert len(self.wav_names) == len(self.wav_paths)
        assert len(self.wav_paths) == len(self.text)

        self.n_per_epoch = len(self.wav_paths)
        self.n_samples_seen_ = 0

        self.buffer_size = 2
        self.minibatch_size = minibatch_size
        self.input_qsize = 5
        self.min_input_qsize = 2
        if len(self.wav_paths) % self.minibatch_size != 0:
            logger.info("WARNING: Sample size not an even multiple of minibatch size")
            logger.info("Truncating...")
            self.wav_paths = self.wav_paths[:-(
                len(self.wav_paths) % self.minibatch_size)]
            self.text = self.text[:-(
                len(self.text) % self.minibatch_size)]

        assert len(self.wav_paths) % self.minibatch_size == 0
        assert len(self.text) % self.minibatch_size == 0

        self.grouped_wav_paths = zip(*[iter(self.wav_paths)] *
                                      self.minibatch_size)
        self.grouped_text = zip(*[iter(self.text)] *
                                     self.minibatch_size)
        assert len(self.grouped_wav_paths) == len(self.grouped_text)
        self._init_queues()

    def _init_queues(self):
        # Infinite...
        self.grouped_elements = itertools.cycle(zip(self.grouped_wav_paths,
                                                self.grouped_text))
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

        for i in range(1):
            self.it = BlizzardThread(self.queue, self.out_queue, self._pre)
            self.it.setDaemon(True)
            self.it.start()

        # Populate queue with some paths to image data
        for n, _ in enumerate(range(self.input_qsize)):
            group = self.grouped_elements.next()
            self.queue.put(group)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def reset(self):
        self.n_samples_seen_ = 0
        self._init_queues()

    def _step(self):
        if self.n_samples_seen_ >= self.n_per_epoch:
            self.reset()
            raise StopIteration("End of epoch")
        wav_group, text_group = self.out_queue.get()
        self.n_samples_seen_ += self.minibatch_size
        if self.queue.qsize() <= self.min_input_qsize:
            for i in range(self.input_qsize):
                group = self.grouped_elements.next()
                self.queue.put(group)
        return wav_group, text_group


def fetch_sample_audio_chords(n_samples=None):
    url = "https://dl.dropboxusercontent.com/u/15378192/piano_chords.tar.gz"
    partial_path = get_resource_dir("chords")
    full_path = os.path.join(partial_path, "piano_chords.tar.gz")
    if not os.path.exists(full_path):
        download(url, full_path)

    tf = tarfile.open(full_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    chords = []
    logger.info("Loading audio files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        chords.append(d)
    return fs, chords, wav_names


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    partial_path = get_resource_dir("fruit")
    full_path = os.path.join(partial_path, "audio.tar.gz")
    if not os.path.exists(full_path):
        download(url, full_path)

    tf = tarfile.open(full_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    logger.info("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech, wav_names


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def _wav2array(nchannels, sampwidth, data):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)

    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
    """
    Read a wav file.

    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.

    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def fetch_sample_speech_ono(n_samples=None):
    datapath = os.path.join("ono_wav", "*wav")
    wav_names = glob.glob(datapath)
    wav_names = [w for w in wav_names
                 if "EKENWAY" in w]
    wav_names = [w for w in wav_names
                 if "PAIN" in w]

    speech = []
    logger.info("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        fs, bitw, d = readwav(wav_name)
        # 24 bit but only 16 used???
        d = d.astype('float32') / (2 ** 15)
        d = sg.decimate(d, 6, ftype="fir")[::6]
        # decimate to 8k
        fs = 8000
        speech.append(d)
    return fs, speech, wav_names


def fetch_sample_speech_walla(n_samples=None):
    datapath = os.path.join("walla_wav", "*wav")
    names = glob.glob(datapath)

    speech = []
    wav_names = []

    logger.info("Loading speech files...")
    for name in names[:n_samples]:
        fs, bitw, d = readwav(name)
        d = d.astype('float32') / (2 ** 15)
        inds = np.arange(0, len(d), 16000)
        for i, j in zip(inds[:-1], inds[1:]):
            dij = d[i:j]
            dij = sg.decimate(dij, 2, ftype="iir")[::2]
            # decimate to 8k
            fs = 8000
            speech.append(dij)
            wav_names.append(name)
        if len(speech) > 200:
           break
    return fs, speech, wav_names


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


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    window_step : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


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


def overlap_add(X_strided, window_step, wsola=False):
    """
    overlap add to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    window_step : int
       step size for overlap add

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    n_rows, window_size = X_strided.shape

    # Start with largest size (no overlap) then truncate after we finish
    # +2 for one window on each side
    X = np.zeros(((n_rows + 2) * window_size,)).astype(X_strided.dtype)
    start_index = 0

    total_windowing_sum = np.zeros((X.shape[0]))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (
        window_size - 1))
    for i in range(n_rows):
        end_index = start_index + window_size
        if wsola:
            offset_size = window_size - window_step
            offset = xcorr_offset(X[start_index:start_index + offset_size],
                                  X_strided[i, :offset_size])
            ss = start_index - offset
            st = end_index - offset
            if start_index - offset < 0:
                ss = 0
                st = 0 + (end_index - start_index)
            X[ss:st] += X_strided[i]
            total_windowing_sum[ss:st] += win
            start_index = start_index + window_step
        else:
            X[start_index:end_index] += X_strided[i]
            total_windowing_sum[start_index:end_index] += win
            start_index += window_step
    # Not using this right now
    #X = np.real(X) / (total_windowing_sum + 1)
    X = X[:end_index]
    return X


def stft(X, fftsize=128, step="half", mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = fftpack.rfft
        cut = -1
    else:
        local_fft = fftpack.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()
    if step == "half":
        X = halfoverlap(X, fftsize)
    else:
        X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, step="half", wsola=False, mean_normalize=True,
          real=False, compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = fftpack.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = fftpack.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2] = X
        X_pad[:, fftsize // 2:] = 0
        X = X_pad
    X = local_ifft(X).astype("float64")
    if step == "half":
        X = invert_halfoverlap(X)
    else:
        X = overlap_add(X, step, wsola=wsola)
    if mean_normalize:
        X -= np.mean(X)
    return X


def mdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    X = halfoverlap(X, N)
    X = sine_window(X)
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # Use transpose due to "samples as rows" convention
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M).T
    return np.dot(X, tf)


def imdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    N_4 = N / 4
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # inverse *is not* transposed
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M)
    X_r = np.dot(X, tf) / N_4
    X_r = sine_window(X_r)
    X = invert_halfoverlap(X_r)
    return X


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
    partial = np.arange(n_filts + 2) / (n_filts + 1.) * (max_mel - min_mel)
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


def sine_window(X):
    """
    Apply a sinusoid window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    i = np.arange(X.shape[1])
    win = np.sin(np.pi * (i + 0.5) / X.shape[1])
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def angle_to_sin_cos(arr_angle):
    return np.hstack((np.sin(arr_angle), np.cos(arr_angle)))


def sin_cos_to_angle(arr_sin, arr_cos):
    return np.arctan2(arr_sin, arr_cos)


def xcorr_offset(x1, x2):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave


def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False,
                               complex_input=False):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print("Runnning iter %i" % i)
        if i == 0 and not complex_input:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        phase = phase[:len(X_s)]
        X_s = X_s[:len(phase)]
        X_best = X_s * phase
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)


def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal

    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html

    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m

    Parameters
    ----------
    X : ndarray
        Raw input signal

    window_size : int, optional (default=256)
        The window size to use, in samples.

    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.

    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.hamming(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        try:
            start = np.max([20, start[0]])
        except IndexError:
            start = 20
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = n_points // window_step
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                       np.zeros((X.shape[0], pad_sizes[1]))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        ((n_windows - 1) * window_step + window_size))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, 2 * window_size - 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        WXX = XX * sg.hanning(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order)]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs
        per_frame_gain[window] = gain
        assign_range = window * window_step + np.arange(window_size)
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[pad_sizes[0]:]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``

    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``

    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi

    magnitudes : ndarray
       Magnitudes of resonant frequencies

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes


def lpc_to_lsf(all_lpc):
    if len(all_lpc.shape) < 2:
        all_lpc = all_lpc[None]
    order = all_lpc.shape[1] - 1
    all_lsf = np.zeros((len(all_lpc), order))
    for i in range(len(all_lpc)):
        lpc = all_lpc[i]
        lpc1 = np.append(lpc, 0)
        lpc2 = lpc1[::-1]
        sum_filt = lpc1 + lpc2
        diff_filt = lpc1 - lpc2

        if order % 2 != 0:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, -1])
            deconv_sum, _ = sg.deconvolve(sum_filt, [1, 1])

        roots_diff = np.roots(deconv_diff)
        roots_sum = np.roots(deconv_sum)
        angle_diff = np.angle(roots_diff[::2])
        angle_sum = np.angle(roots_sum[::2])
        lsf = np.sort(np.hstack((angle_diff, angle_sum)))
        if len(lsf) != 0:
            all_lsf[i] = lsf
    return np.squeeze(all_lsf)


def lsf_to_lpc(all_lsf):
    if len(all_lsf.shape) < 2:
        all_lsf = all_lsf[None]
    order = all_lsf.shape[1]
    all_lpc = np.zeros((len(all_lsf), order + 1))
    for i in range(len(all_lsf)):
        lsf = all_lsf[i]
        zeros = np.exp(1j * lsf)
        sum_zeros = zeros[::2]
        diff_zeros = zeros[1::2]
        sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
        diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
        sum_filt = np.poly(sum_zeros)
        diff_filt = np.poly(diff_zeros)

        if order % 2 != 0:
            deconv_diff = sg.convolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff = sg.convolve(diff_filt, [1, -1])
            deconv_sum = sg.convolve(sum_filt, [1, 1])

        lpc = .5 * (deconv_sum + deconv_diff)
        # Last coefficient is 0 and not returned
        all_lpc[i] = lpc[:-1]
    return np.squeeze(all_lpc)


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html

    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients

    per_frame_gain : ndarray
        Gain coefficients

    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise

    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows

    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.hanning(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.hanning(window_size) * newbit
    synthesized = sg.lfilter([1], [1, -emphasis], synthesized)
    return synthesized


class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 randomize=False,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.randomize = randomize
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)
        self.one_hot_class_size = one_hot_class_size
        self.random_state = np.random.RandomState(2017)
        len0 = len(list_of_containers[0])
        assert all([len(ci) == len0 for ci in list_of_containers])
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)

    def reset(self):
        self.slice_start_ = self.start_index
        if self.randomize:
            start_ind = self.start_index
            stop_ind = min(len(self.list_of_containers[0]), self.stop_index)
            inds = np.arange(start_ind, stop_ind).astype("int32")
            # If start index is > 0 then pull some mad hackery to only shuffle
            # the end part - eg. validation set.
            self.random_state.shuffle(inds)
            if start_ind > 0:
                orig_inds = np.arange(0, start_ind).astype("int32")
                inds = np.concatenate((orig_inds, inds))
            new_list_of_containers = []
            for ci in self.list_of_containers:
                nci = [ci[i] for i in inds]
                if isinstance(ci, np.ndarray):
                    nci = np.array(nci)
                new_list_of_containers.append(nci)
            self.list_of_containers = new_list_of_containers

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = slice(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        sliced_c = []
        for c in self.list_of_containers:
            slc = c[ind]
            arr = np.asarray(slc)
            sliced_c.append(arr)
        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
            if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                maxlen = max([len(i) for i in sc])
                # Assume they at least have the same internal dtype
                if len(sc[0].shape) > 1:
                    total_shape = (maxlen, sc[0].shape[1])
                elif len(sc[0].shape) == 1:
                    total_shape = (maxlen, 1)
                else:
                    raise ValueError("Unhandled array size in list")
                if self.axis == 0:
                    raise ValueError("Unsupported axis of iteration")
                    new_sc = np.zeros((len(sc), total_shape[0],
                                       total_shape[1]))
                    new_sc = new_sc.squeeze().astype(sc[0].dtype)
                else:
                    new_sc = np.zeros((total_shape[0], len(sc),
                                       total_shape[1]))
                    new_sc = new_sc.astype(sc[0].dtype)
                    for m, sc_i in enumerate(sc):
                        new_sc[:len(sc_i), m, :] = sc_i
                sliced_c[n] = new_sc
            else:
                # Hit this case if all sequences are the same length
                if self.axis == 1:
                    sliced_c[n] = sc.transpose(1, 0, 2)
        return sliced_c

    def _slice_with_masks(self, ind):
        cs = self._slice_without_masks(ind)
        if self.axis == 0:
            ms = [np.ones_like(c[:, 0]) for c in cs]
            raise ValueError("NYI - see axis=0 case for ideas")
            sliced_c = []
            for n, c in enumerate(self.list_of_containers):
                slc = c[ind]
                for ii, si in enumerate(slc):
                    ms[n][ii, len(si):] = 0.
        elif self.axis == 1:
            ms = [np.ones_like(c[:, :, 0]) for c in cs]
            sliced_c = []
            for n, c in enumerate(self.list_of_containers):
                slc = c[ind]
                for ii, si in enumerate(slc):
                    ms[n][len(si):, ii] = 0.
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]


class audio_file_iterator(object):
    def __init__(self, file_glob, minibatch_size, start_index=0,
                 stop_index=np.inf, make_mask=True,
                 randomize=True, preprocess=None,
                 preprocess_kwargs={}):
        """
        Supports regular int, negative indexing, or float for setting
        stop_index
        """
        self.minibatch_size = minibatch_size
        self.file_list = glob.glob(file_glob)
        if len(self.file_list) == 0:
            raise ValueError("Invalid file glob, no files found!")
        if randomize:
            self.random_state = np.random.RandomState(2177)
        self.random_state.shuffle(self.file_list)
        self.make_mask = make_mask
        ext = self.file_list[0].split(".")[-1]
        if ext == "flac":
            import scikits.audiolab
            def _read(fpath):
                # d in -1, 1
                d, fs, enc = scikits.audiolab.flacread(fpath)
                return d
        else:
            raise ValueError("Unhandled extension %s" % ext)
        self._read_file = _read
        if preprocess is None:
            self.preprocess_function = lambda x: x[:, None]
        elif preprocess == "quantize_window":
            # reslice to 2D in sets of 128, 4
            n_frame = 4
            n_bins = 256
            if "n_frame" in preprocess_kwargs.keys():
                n_frame = preprocess_kwargs["n_frame"]

            if "n_bins" in preprocess_kwargs.keys():
                n_bins = preprocess_kwargs["n_bins"]
            def p(x):
                # apply_ functions meant to operate on lists of sequences
                return apply_quantize_window_preproc([x], n_bins, n_frame,
                                                     mn=-1, mx=1)[0].astype("float32")
            self.preprocess_function = p
        elif preprocess == "quantize":
            n_bins = 256
            if "n_bins" in preprocess_kwargs.keys():
                n_bins = preprocess_kwargs["n_bins"]
            def p(x):
                # apply_ functions meant to operate on lists of sequences
                return apply_quantize_preproc([x], n_bins,
                                              mn=-1, mx=1)[0][:, None].astype("float32")
            self.preprocess_function = p


        if stop_index >= 1:
            self.stop_index = int(min(stop_index, len(self.file_list)))
        elif stop_index > 0:
            # percentage
            self.stop_index = int(stop_index * len(self.file_list))
        elif stop_index < 0:
            # negative index - must be int!
            self.stop_index = len(self.file_list) + int(stop_index)

        if start_index < 0:
            # negative indexing
            self.start_index = len(self.file_list) + start_index
        elif start_index < 1:
            # float
            self.start_index = int(start_index * len(self.file_list))
        else:
            # regular
            self.start_index = int(start_index)
        if self.start_index >= self.stop_index:
            ss = "Invalid indexes - stop "
            ss += "%s <= start %s !" % (self.stop_index, self.start_index)
            raise ValueError(ss)
        self._current_index = self.start_index

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        s = self._current_index
        e = self._current_index + self.minibatch_size
        if e > self.stop_index:
            raise StopIteration("End of audio file iterator reached!")
        files_to_get = self.file_list[s:e]
        data = [self._read_file(fp) for fp in files_to_get]
        data = [self.preprocess_function(d) for d in data]

        li = list_iterator([data], self.minibatch_size, axis=1, start_index=0,
                           stop_index=len(data), make_mask=self.make_mask)
        res = next(li)
        self._current_index = e
        return res

    def reset(self):
        self._current_index = self.start_index


def get_dataset_dir(dataset_name):
    """ Get dataset directory path """
    return os.sep.join(os.path.realpath(__file__).split
                       (os.sep)[:-1] + [dataset_name])


def numpy_one_hot(labels_dense, n_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dtype = labels_dense.dtype
    labels_dense = labels_dense.ravel().astype("int32")
    n_labels = labels_dense.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot[np.arange(n_labels).astype("int32"),
                   labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(n_classes,))
    return labels_one_hot.astype(labels_dtype)


def tokenize_ind(phrase, vocabulary):
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = numpy_one_hot(phrase, vocabulary_size)
    return phrase


def apply_quantize_preproc(X, n_bins=256, mn=-32768, mx=32768):
    def scale(x):
        x = (x - mn) / (mx - mn)
        return x

    # Extra n because bin is reserved kwd in Python
    bins = np.linspace(0, 1, n_bins)
    def binn(x):
        shp = x.shape
        return (np.digitize(x.ravel(), bins) - 1).reshape(shp)
    X = [scale(Xi) for Xi in X]
    X = [binn(Xi) for Xi in X]
    return X


def apply_quantize_window_preproc(X, n_bins=256, n_frame=4, mn=-32768, mx=32768):
    def scale(x):
        # scale to 0, 1
        x = a.astype("float64")
        x = (x - x.min()) / (x.max() - x.min())
        return x

    # Extra n because bin is reserved kwd in Python
    bins = np.linspace(0, 1, n_bins)
    def binn(x):
        x = x * (bins - 0.5) # 0 to 255.5
        x = x.astype("int32").astype("float32")
        return x
    X = [scale(Xi) for Xi in X]
    X = [binn(Xi) for Xi in X]

    def _s(x):
        xi = x[:len(x) - len(x) % n_frame]
        # chunk it
        xi = xi.reshape((-1, n_frame))
        #xi = overlap(xi, 4, 1)
        return xi

    X_windows = []
    for x in X:
        xw = _s(x)
        X_windows.append(xw)
    X = X_windows
    return X


def apply_stft_preproc(X, n_fft=128, n_step_frac=4):
    n_step = n_fft // n_step_frac

    def _pre(x):
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    X = [_pre(Xi) for Xi in X]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x

    X = [scale(Xi) for Xi in X]

    def _re(x):
        X_mag_phase = unscale(x)
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r
    return X, _re


def apply_binned_spectrogram_preproc(X, n_fft=512, n_step_frac=10,
                                     n_bins=10):
    n_step = n_fft // n_step_frac

    def _pre(x):
        X_mag = np.abs(stft(x, n_fft, step=n_step))
        X_mag = np.log10(X_mag + 1E-9)
        return X_mag

    X = [_pre(Xi) for Xi in X]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len
    X_max = np.max([np.max(Xi) for Xi in X])
    X_min = np.min([np.min(Xi) for Xi in X])

    def scale(x):
        x = (x - X_min) / (X_max - X_min)
        return x

    def unscale(x):
        x = x * (X_max - X_min) + X_min
        return x

    # Extra n because bin is reserved kwd in Python
    bins = np.linspace(0, 1, n_bins)
    def binn(x):
        shp = x.shape
        return (np.digitize(x.ravel(), bins) - 1).reshape(shp)

    def unbin(x):
        # use middle instead of left edge?
        return x / float(n_bins)

    X = [scale(Xi) for Xi in X]
    X = [binn(Xi) for Xi in X]

    """
    import matplotlib.pyplot as plt
    plt.matshow(X[0][::-1, ::-1].T)
    plt.matshow(unbin(X[0])[::-1, ::-1].T)
    from IPython import embed; embed()
    raise ValueError()
    """

    def _re(x):
        X_ub = unbin(x)
        X_mag = unscale(X_ub)
        X_mag = 10 ** X_mag
        X_s = np.hstack((X_mag, X_mag[:, ::-1]))
        X_r = iterate_invert_spectrogram(X_s, n_fft, n_step)
        return X_r
    return X, _re


def apply_labeled_spectrogram_preproc(X, n_fft=512, n_step_frac=10,
                                      n_bins=10):

    X, _re = apply_binned_spectrogram_preproc(X, n_fft=n_fft,
                                              n_step_frac=n_step_frac,
                                              n_bins=n_bins)
    # Make sum and var lookup for discretization of class preds
    sum_lookup = defaultdict(list)
    var_lookup = defaultdict(list)
    sum_labels = []
    var_labels = []
    for Xi in X:
        sums = Xi.sum(axis=1)
        sum_label_set = []
        for n, s in enumerate(sums):
            sum_lookup[s].append(Xi[n])
            sum_label_set.append(s)
        sum_labels.append(sum_label_set)

        # vars is reserved...
        varis = Xi.var(axis=1)
        var_label_set = []
        for n, v in enumerate(varis):
            var_lookup[v].append(Xi[n])
            var_label_set.append(v)
        var_labels.append(var_label_set)

    class_to_sum = {k: v for k,v in enumerate(sorted(sum_lookup.keys()))}
    sum_to_class = {v: k for k,v in class_to_sum.items()}
    class_to_var = {k: v for k,v in enumerate(sorted(var_lookup.keys()))}
    var_to_class = {v: k for k,v in class_to_var.items()}

    sum_labels = [np.array([sum_to_class[ii] for ii in i]).astype("int32")
                  for i in sum_labels]
    var_labels = [np.array([var_to_class[ii] for ii in i]).astype("int32")
                  for i in var_labels]

    def _full_re(s, v):
        assert len(s) == len(v)
        # slow distance minimization loop for decode
        res = []
        for s_t, v_t in zip(s, v):
            s_cand = sum_lookup[class_to_sum[s_t]]
            v_cand = var_lookup[class_to_var[v_t]]
            dist = np.zeros((len(s_cand), len(v_cand)))
            # Could do as as outer product for speed
            for i, s_c in enumerate(s_cand):
                s_c_norm = np.sqrt(np.sum(s_c ** 2))
                for j, v_c in enumerate(v_cand):
                    v_c_norm = np.sqrt(np.sum(v_c ** 2))
                    dist_ij = np.sqrt(np.sum((s_c - v_c) ** 2))
                    dist_ij /= (s_c_norm * v_c_norm)
                    dist[i, j] = dist_ij
            if any(dist.ravel() < 1E-9):
                # if any are almost 0, take it!
                sum_idx = np.where(dist < 1E-9)[0][0]
            else:
                # find which one minimizes all var scores
                sum_idx = np.argmin(np.sum(dist, axis=1))
            final_cand = s_cand[sum_idx]
            res.append(final_cand)
        res = np.asarray(res)
        res = _re(res)
        return res
    return sum_labels, var_labels, _full_re


def apply_lpc_softmax_preproc(X, fs=8000):
    # 256 @ 8khz - .032
    ws = 2 ** int(np.log(0.032 * fs) / np.log(2))
    window_size = ws
    window_step = int(.2 * window_size)
    lpc_order = 12
    def _pre(x):
        a, g, e = lpc_analysis(x, order=lpc_order, window_step=window_step,
                               window_size=window_size, emphasis=0.9,
                               copy=True)
        a = lpc_to_lsf(a)
        f_sub = np.hstack((a, g))
        v, p = voiced_unvoiced(x, window_size=window_size,
                               window_step=window_step)
        cut_len = e.shape[0] - e.shape[0] % len(a)
        e = e[:cut_len]
        e = e.reshape((len(a), -1))
        f_full = np.hstack((a, g, v, e))
        return f_sub, f_full

    X = [_pre(Xi)[0] for Xi in X]
    X_stack = np.vstack(X)
    kmeans_results = []
    random_state = np.random.RandomState(1999)
    n_clust = 60
    for dim in range(X_stack.shape[1]):
        print("Processing dim %i of %i" % (dim + 1, X_stack.shape[1]))
        # Assume some clusters will die
        res = kmeans(X_stack[:, dim], n_clust * 2)
        sub = list(range(len(res[0])))
        random_state.shuffle(sub)
        assert len(sub) > n_clust
        kmeans_results.append(res[0][sub[:n_clust]])

    def _vq(Xi):
        Xi2 = Xi.copy()
        for dim in range(X_stack.shape[1]):
            idx, _ = vq(Xi[:, dim], kmeans_results[dim])
            Xi2[:, dim] = idx
        return Xi2

    def _unvq(Xi):
        """
        assumes vq indices have been cast to float32
        """
        Xi2 = Xi.copy()
        for dim in range(X_stack.shape[1]):
            Xi2[:, dim] = kmeans_results[dim][Xi[:, dim].astype("int32")]
        return Xi2

    X = [_vq(Xi) for Xi in X]

    def _apply(Xi):
        return _vq(_pre(Xi)[0])

    def _re_sub(sub):
        sub = _unvq(sub)
        a = sub[:, :-1]
        a = lsf_to_lpc(a)
        g = sub[:, -1:]
        x_r = lpc_synthesis(a, g, emphasis=0.9,
                            window_step=window_step)
        agc_x_r, _, _ = time_attack_agc(x_r, fs)
        return agc_x_r

    def _re_full(full):
        raise ValueError("NYI")
        a = full[:, :lpc_order]
        a = np.hstack(np.ones_like(a[:, 0]), a)
        offset = lpc_order
        g = full[:, offset:offset + 1]
        offset = offset + 1
        v = full[:, offset:offset + 1]
        offset = offset + 1
        e = full[:, offset:].ravel()
        x_r = lpc_synthesis(a, g, e, voiced_frames=v,
                            emphasis=0.9, window_step=window_step)
        agc_x_r, _, _ = time_attack_agc(x_r, fs)
        return agc_x_r

    return X, _apply, _re_sub


def fetch_binarized_mnist():
    # public version
    url = 'https://github.com/mgermain/MADE/releases/download/ICML2015/'
    url += 'binarized_mnist.npz'
    partial_path = get_resource_dir("binarized_mnist")
    fname = "binarized_mnist.npz"
    full_path = os.path.join(partial_path, fname)
    if not os.path.exists(full_path):
        download(url, full_path)
    d = np.load(full_path)
    train = d["train_data"]
    valid = d["valid_data"]
    test = d["test_data"]
    all_ = np.concatenate((train, valid, test), axis=0)
    train_indices = np.arange(train.shape[0])
    valid_indices = train.shape[0] + np.arange(valid.shape[0])
    test_indices = train.shape[0] + valid.shape[0] + np.arange(test.shape[0])
    t = {}
    t["data"] = all_.reshape((-1, 28, 28))
    t["train_indices"] = train_indices
    t["valid_indices"] = valid_indices
    t["test_indices"] = test_indices
    """
    # personal version
    url = "https://dl.dropboxusercontent.com/u/15378192/binarized_mnist_%s.npy"
    fname = "binarized_mnist_%s.npy"
    for s in ["train", "valid", "test"]:
        full_path = os.path.join(partial_path, fname % s)
        if not os.path.exists(partial_path):
            os.makedirs(partial_path)
        if not os.path.exists(full_path):
            download(url % s, full_path, progress_update_percentage=1)
    """
    return t


def fetch_nottingham():
    midiread, midiwrite = midiwrap()
    url = "http://www.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip"
    partial_path = get_resource_dir("nottingham")
    full_path = os.path.join(partial_path, "Nottingham.zip")
    if not os.path.exists(full_path):
        download(url, full_path)
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(partial_path)
        zip_ref.close()
    key_range = (21, 109)
    dt = 0.3

    all_data = []
    with zipfile.ZipFile(full_path, "r") as f:
        for name in f.namelist():
            if ".mid" not in name:
                # Skip README
                continue
            p = os.path.join(partial_path, name)
            data = midiread(p, key_range, dt).piano_roll.astype(
                theano.config.floatX)
            all_data.append(data)
    raise ValueError("Fix to return dictionary like the rest of the fetch_")
    return key_range, dt, all_data


def fetch_fruitspeech_softmax():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    # Is it kosher to kmeans on all the data?
    X, _apply, _re = apply_lpc_softmax_preproc(d)

    """
    for n, Xi in enumerate(X[::8]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech_spectrogram():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    X, _re = apply_binned_spectrogram_preproc(d)

    """
    for n, Xi in enumerate(X[::8]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech():
    #258 to get 256 in fancy log
    #n_bins = 258
    n_bins = 256

    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    X = d
    # DC center
    X = [di - di.mean() for di in d]
    X_max = np.max([np.max(Xi) for Xi in X]).astype("float32")
    X_min = np.min([np.min(Xi) for Xi in X]).astype("float32")

    def scale(x):
        x = (x - X_min) / (X_max - X_min)
        return x

    def unscale(x):
        x = x * (X_max - X_min) + X_min
        return x

    # Extra n because bin is reserved kwd in Python
    #bins = np.linspace(0, 1, n_bins)
    # 9 is 10 - 1, 10 ** 1
    """
    bins = (np.logspace(0, 1, n_bins // 2) - 1) / 9
    bins2 = bins / 2. + .5
    bins = .5 - bins[::-1] / 2.
    #want 256 bins - end up making a slightly larger bin above .5 (0 in -1, 1)
    bins = np.concatenate((bins, bins2[2:]))
    """
    bins = np.linspace(0, 1, n_bins)
    def binn(x):
        shp = x.shape
        return (np.digitize(x.ravel(), bins) - 1).reshape(shp)

    def unbin(x):
        # use middle instead of left edge?
        return np.array([bins[xi] for xi in x]).astype("float32")

    X = [scale(Xi) for Xi in X]
    X = [binn(Xi) for Xi in X]

    def _re(x):
        X_ub = unbin(x)
        X_r = unscale(X_ub)
        return X_r

    """
    for n, Xi in enumerate(X[::8]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech_spectral():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    n_fft = 128
    n_step = n_fft // 4

    def _pre(x):
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    X = [_pre(di) for di in d]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x

    X = [scale(Xi) for Xi in X]

    def _re(x):
        X_mag_phase = unscale(x)
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r

    """
    for n, Xi in enumerate(X[::8]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech_spectrogram_nonpar():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    data1, data2, _re = apply_labeled_spectrogram_preproc(d, n_bins=10)

    """
    # Check reconstructions
    for n, (s_l, v_l) in enumerate(list(zip(data1, data2))[::8]):
        di = _re(s_l, v_l)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))
    raise ValueError()
    """

    data1_size = max([max(d1) for d1 in data1])
    data2_size = max([max(d2) for d2 in data2])
    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data1"] = data1
    speech["data1_size"] = data1_size
    speech["data2"] = data2
    speech["data2_size"] = data2_size
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_fruitspeech_nonpar():
    fs, d, wav_names = fetch_sample_speech_fruit()
    def matcher(name):
        return name.split("/")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    # 256 @ 8khz - .032
    ws = 2 ** int(np.log(0.032 * fs) / np.log(2))
    window_size = ws
    window_step = int(.15 * window_size)
    lpc_order = 30
    def _pre(x):
        a, g, e = lpc_analysis(x, order=lpc_order, window_step=window_step,
                               window_size=window_size, emphasis=0.9,
                               copy=True)
        f_sub = a[:, 1:]
        f_full = stft(x, window_size, window_step) #, compute_onesided=False)
        """
        v, p = voiced_unvoiced(x, window_size=window_size,
                               window_step=window_step)
        cut_len = e.shape[0] - e.shape[0] % len(a)
        e = e[:cut_len]
        e = e.reshape((len(a), -1))
        f_full = np.hstack((a, g, v, e))
        """
        if len(f_sub) >= len(f_full):
            f_sub = f_sub[:len(f_full)]
        else:
            f_full = f_full[:len(f_sub)]
        return f_sub, f_full

    def _train(list_of_data):
        f_sub = None
        f_full = None
        for i in range(len(list_of_data)):
            f_sub_i, f_full_i = _pre(list_of_data[i])
            if f_sub is None:
                f_sub = f_sub_i
                f_full = f_full_i
            else:
                f_sub = np.vstack((f_sub, f_sub_i))
                if f_full.shape[1] > f_full_i.shape[1]:
                    f_full_i = np.hstack(
                        (f_full_i, np.zeros_like(f_full_i[:, -1][:, None])))
                elif f_full_i.shape[1] > f_full.shape[1]:
                    f_full_i = f_full_i[:, :f_full.shape[1]]
                f_full = np.vstack((f_full, f_full_i))
        sub_clusters = f_sub
        full_clusters = f_full
        return sub_clusters, full_clusters

    def _clust(x, sub_clusters, extras=None):
        f_sub, f_full = _pre(x)
        f_clust = f_sub
        mem, _ = vq(copy.deepcopy(f_clust), copy.deepcopy(sub_clusters))
        # scipy vq sometimes puts out garbage? choose one at random...
        # possibly related to NaN in input
        #mem[np.abs(mem) >= len(mel_clusters)] = mem[
        #    np.abs(mem) >= len(mel_clusters)] % len(mel_clusters)
        return mem

    def _re(x, sub_clusters, full_clusters):
        memberships = x
        vq_x = full_clusters[memberships]
        """
        # STFT frames not working well in rec
        x_r = iterate_invert_spectrogram(vq_x, window_size, window_step,
                                         n_iter=50, complex_input=True)
        """
        x_r = istft(vq_x, window_size, window_step, wsola=True)
        """
        a = vq_x[:, :lpc_order + 1]
        offset = lpc_order + 1
        g = vq_x[:, offset:offset + 1]
        offset = offset + 1
        v = vq_x[:, offset:offset + 1]
        offset = offset + 1
        e = vq_x[:, offset:].ravel()
        x_r = lpc_synthesis(a, g, e, voiced_frames=v,
                            emphasis=0.9, window_step=window_step)
        """
        agc_x_r, _, _ = time_attack_agc(x_r, fs)
        return agc_x_r

    random_state = np.random.RandomState(1999)
    all_ind = list(range(8))
    # Get 5 random subsets
    random_state.shuffle(all_ind)
    ind = all_ind[:6]

    d1 = []
    for i in ind:
        d1 += d[i::8]

    sub_clusters, full_clusters = _train(d1)


    def _re_wrap(x):
        x = x.argmax(axis=-1)
        re_d = _re(x, sub_clusters, full_clusters)
        return re_d

    def _apply(x):
        m = _clust(x, sub_clusters)
        return m

    X = [_apply(Xi) for Xi in d]
    X = [numpy_one_hot(Xi, len(sub_clusters)) for Xi in X]

    """
    for n, Xi in enumerate(X[all_ind[0]::8]):
        di = _re_wrap(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    for n, Xi in enumerate(X[all_ind[-1]::8]):
        di = _re_wrap(Xi)
        wavfile.write("to_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re_wrap
    return speech


def fetch_ono():
    fs, d, wav_names = fetch_sample_speech_ono()
    # Force 1D
    d = [di.squeeze() for di in d]
    def matcher(name):
        return name.split("PAIN")[1].split("_")[1]

    classes = [matcher(wav_name) for wav_name in wav_names]
    low = ["L1", "L2", "L3"]
    uw = ["U1", "U2", "U3", "UF"]
    high = ["L4", "L5", "SA"]
    final_classes = []
    for c in classes:
        if c in low:
            final_classes.append("low")
        elif c in high:
            final_classes.append("hi")
        elif c in uw:
            final_classes.append("uw")
        else:
            raise ValueError("Unknown class %s" % c)
    classes = final_classes
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    n_fft = 128
    n_step = n_fft // 4

    def _pre(x):
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    X = [_pre(di) for di in d]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x

    X = [scale(Xi) for Xi in X]

    def _re(x):
        X_mag_phase = unscale(x)
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r

    """
    for n, Xi in enumerate(X[:10]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech


def fetch_walla():
    fs, d, wav_names = fetch_sample_speech_walla()
    # Force 1D
    d = [di.squeeze() for di in d]
    classes = ["ab" for wav_name in wav_names]
    all_chars = [c for c in sorted(list(set("".join(classes))))]
    char2code = {v: k for k, v in enumerate(all_chars)}
    vocabulary_size = len(char2code.keys())
    y = []
    for n, cl in enumerate(classes):
        y.append(tokenize_ind(cl, char2code))

    n_fft = 128
    n_step = n_fft // 4

    def _pre(x):
        X_stft = stft(x, n_fft, step=n_step)
        # Power spectrum
        X_mag = complex_to_abs(X_stft)
        X_mag = np.log10(X_mag + 1E-9)
        # unwrap phase then take delta
        X_phase = complex_to_angle(X_stft)
        X_phase = np.vstack((np.zeros_like(X_phase[0][None]), X_phase))
        # Adding zeros to make network predict what *delta* in phase makes sense
        X_phase_unwrap = np.unwrap(X_phase, axis=0)
        X_phase_delta = X_phase_unwrap[1:] - X_phase_unwrap[:-1]
        X_mag_phase = np.hstack((X_mag, X_phase_delta))
        return X_mag_phase

    X = [_pre(di) for di in d]

    X_len = np.sum([len(Xi) for Xi in X])
    X_sum = np.sum([Xi.sum(axis=0) for Xi in X], axis=0)
    X_mean = X_sum / X_len
    X_var = np.sum([np.sum((Xi - X_mean[None]) ** 2, axis=0)
                    for Xi in X], axis=0) / X_len

    def scale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = (_x - _mean) / _var
        return x

    def unscale(x):
        # WARNING: OPERATES IN PLACE!!!
        # Can only realistically scale magnitude...
        # Phase cost depends on circularity
        x = np.copy(x)
        _x = x[:, :n_fft // 2]
        _mean = X_mean[None, :n_fft // 2]
        _var = X_var[None, :n_fft // 2]
        x[:, :n_fft // 2] = _x * _var + _mean
        return x

    X = [scale(Xi) for Xi in X]

    def _re(x):
        X_mag_phase = unscale(x)
        X_mag = X_mag_phase[:, :n_fft // 2]
        X_mag = 10 ** X_mag
        X_phase_delta = X_mag_phase[:, n_fft // 2:]
        # Append leading 0s for consistency
        X_phase_delta = np.vstack((np.zeros_like(X_phase_delta[0][None]),
                                   X_phase_delta))
        X_phase = np.cumsum(X_phase_delta, axis=0)[:-1]
        X_stft = abs_and_angle_to_complex(X_mag, X_phase)
        X_r = istft(X_stft, n_fft, step=n_step, wsola=False)
        return X_r

    """
    for n, Xi in enumerate(X[:10]):
        di = _re(Xi)
        wavfile.write("t_%i.wav" % n, fs, soundsc(di))

    raise ValueError()
    """

    speech = {}
    speech["vocabulary_size"] = vocabulary_size
    speech["vocabulary"] = char2code
    speech["sample_rate"] = fs
    speech["data"] = X
    speech["target"] = y
    speech["reconstruct"] = _re
    return speech
"""
end datasets
"""

"""
begin initializers and Theano functions
"""
def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros

    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize

    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype(theano.config.floatX)


def np_ones(shape):
    """
    Builds a numpy variable filled with ones

    Parameters
    ----------
    shape, tuple of ints
        shape of ones to initialize

    Returns
    -------
    initialized_ones, array-like
        Array-like of ones the same size as shape parameter
    """
    return np.ones(shape).astype(theano.config.floatX)


def np_uniform(shape, random_state, scale=0.08):
    """
    Builds a numpy variable filled with uniform random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 0.08)
        scale to apply to uniform random values from (-1, 1)
        default of 0.08 results in uniform random values in (-0.08, 0.08)

    Returns
    -------
    initialized_uniform, array-like
        Array-like of uniform random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    # Make sure bounds aren't the same
    return random_state.uniform(low=-scale, high=scale, size=shp).astype(
        theano.config.floatX)


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01

    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype(theano.config.floatX)


def np_tanh_fan_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal uniform random values
        with sqrt(6 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    # The . after the 6 is critical! shape has dtype int...
    bound = scale * np.sqrt(6. / kern_sum)
    return random_state.uniform(low=-bound, high=bound,
                                size=shp).astype(theano.config.floatX)


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype(theano.config.floatX)


def np_sigmoid_fan_uniform(shape, random_state, scale=4.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 4.)
        default of 4. results in uniform random values
        with 4 * sqrt(6 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    return scale * np_tanh_fan_uniform(shape, random_state)


def np_sigmoid_fan_normal(shape, random_state, scale=4.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 4.)
        default of 4. results in normal random values
        with 4 * sqrt(2 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    return scale * np_tanh_fan_normal(shape, random_state)


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale

    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller

    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / kern_sum)  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        theano.config.floatX)


def np_variance_scaled_randn(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with 1 * sqrt(1 / (n_dims)) scale

    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    # Make sure bounds aren't the same
    std = scale * np.sqrt(1. / kern_sum)
    return std * random_state.randn(*shp).astype(theano.config.floatX)


def np_deep_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(6 / (n_dims)) scale

    Returns
    -------
    initialized_deep, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Diving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
        K. He, X. Zhang, S. Ren, J. Sun
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(6. / kern_sum)  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        theano.config.floatX)


def np_deep_scaled_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with 1 * sqrt(2 / (n_dims)) scale

    Returns
    -------
    initialized_deep, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Diving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
        K. He, X. Zhang, S. Ren, J. Sun
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    # Make sure bounds aren't the same
    std = scale * np.sqrt(2. / kern_sum)  # sqrt(3) for std of uniform
    return std * random_state.randn(*shp).astype(theano.config.floatX)


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.

    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prd(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype(theano.config.floatX)


def np_identity(shape, random_state, scale=0.98):
    """
    Identity initialization for square matrices

    Parameters
    ----------
    shape, tuple of ints
        shape of resulting array - shape[0] and shape[1] must match

    random_state, numpy.random.RandomState() object

    scale, float (default 0.98)
        default of .98 results in .98 * eye initialization

    Returns
    -------
    initialized_identity, array-like
        identity initialized square matrix same size as shape

    References
    ----------
    A Simple Way To Initialize Recurrent Networks of Rectified Linear Units
        Q. Le, N. Jaitly, G. Hinton
    """
    assert shape[0] == shape[1]
    res = np.eye(shape[0])
    return (scale * res).astype(theano.config.floatX)


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    if axis < 0:
        axis = tensor_list[0].ndim + axis
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]
    return out


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if type(arr) in [float, int]:
        if name is not None:
            return theano.shared(np.cast[theano.config.floatX](arr))
        else:
            return theano.shared(np.cast[theano.config.floatX](arr), name=name)
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def apply_shared(list_of_numpy):
    return [as_shared(arr) for arr in list_of_numpy]


def make_numpy_biases(bias_dims):
    return [np_zeros((dim,)) for dim in bias_dims]


def make_biases(bias_dims):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element returned!
    blah, = make_biases(...)
    or
    [blah] = make_biases(...)
    """
    bs = make_numpy_biases(bias_dims)
    return apply_shared(bs)


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default"):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element returned!
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init == None:
            if in_dim == out_dim:
                ff[i] = np_ortho
            else:
                ff[i] = np_variance_scaled_uniform
        elif init == "normal":
            ff[i] = np_normal
        elif init == "fan":
            ff[i] = np_tanh_fan_normal
        elif init == "ortho":
            ff[i] = np_ortho
        else:
            raise ValueError("Unknown init type %s" % init)
    if scale == "default":
        ws = [ff[i]((in_dim, out_dim), random_state)
              for i, out_dim in enumerate(out_dims)]
    else:
        ws = [ff[i]((in_dim, out_dim), random_state, scale=scale)
              for i, out_dim in enumerate(out_dims)]
    return ws


def make_weights(in_dim, out_dims, random_state, init=None,
                 scale="default"):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element returned!
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ws = make_numpy_weights(in_dim, out_dims, random_state, init=init,
                            scale=scale)
    return apply_shared(ws)


def LearnedInitHidden(list_of_inputs, list_of_shapes):
    # Helper to allow switch for learned hidden inits
    ret = []
    assert len(list_of_inputs) == len(list_of_shapes)
    for i, shp in enumerate(list_of_shapes):
        name = None
        s = param(name, make_numpy_biases([shp[1]])[0])
        ss = s[None, :] * tensor.ones((shp[0], shp[1]))
        init = theano.ifelse.ifelse(tensor.abs_(ss.sum()) < 1E-12,
                                    ss, list_of_inputs[i])
        ret.append(init)
    return ret


def Embedding(indices, n_symbols, output_dim, random_state, name=None):
    """
    Last dimension of indices tensor must be 1!!!!
    """
    vectors = param(name,
        random_state.randn(
            n_symbols,
            output_dim
        ).astype(theano.config.floatX)
    )
    ii = indices.astype("int32")
    output_shape = [
        ii.shape[i]
        for i in range(ii.ndim - 1)
    ] + [output_dim]
    return vectors[ii.flatten()].reshape(output_shape)


def Linear(list_of_inputs, input_dims, output_dim, random_state, name=None,
           init=None, scale="default", weight_norm=None, biases=True):
    """
    Can pass weights and biases directly if needed through init
    """
    if weight_norm is None:
        # Let other classes delegate to default of linear
        weight_norm = True
    input_var = tensor.concatenate(list_of_inputs, axis=-1)
    input_dim = sum(input_dims)
    terms = []
    if (init is None) or (type(init) is str):
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        weight_values = init[0]
    weight = param((name, "W"), weight_values)
    # From Ishaan G.
    # http://arxiv.org/abs/1602.07868
    if weight_norm:
        norm_values = np.linalg.norm(weight_values, axis=0)
        norms = param((name, "wn"), norm_values)
        normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
        terms.append(tensor.dot(input_var, normed_weight))
    else:
        terms.append(tensor.dot(input_var, weight))

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim])
        else:
            b = init[1]
        terms.append(param((name, "b"), b))
    out = reduce(lambda a, b: a + b, terms)
    out.name = get_generic_name() + ".output"
    return out


def make_conv_weights(in_dim, out_dims, kernel_size, random_state):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element returned!
    blah, = make_conv_weights(...)
    or
    [blah] = make_conv_weights(...)
    """
    return apply_shared([np_tanh_fan_normal(
        ((in_dim, kernel_size[0], kernel_size[1]),
         (out_dim, kernel_size[0], kernel_size[1])), random_state)
                         for out_dim in out_dims])


def conv2d(input, filters, biases=None, border_mode=0, stride=(1, 1)):
    """
    Light wrapper around conv2d - optionally handle biases
    """
    r = nnet.conv2d(
            input=input,
            filters=filters,
            border_mode=border_mode,
            subsample=stride,
            filter_flip=True)
    if biases is None:
        return r
    else:
        return r + biases.dimshuffle('x', 0, 'x', 'x')


def unpool(input, pool_size=(1, 1)):
    """
    Repeat unpooling
    """
    return input.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3)


def conv2d_transpose(input, filters, biases=None, border_mode=0, stride=(1, 1)):
    """
    Light wrapper around conv2d_transpose
    """
    # swap to in dim out dim to make life easier
    filters = filters.transpose(1, 0, 2, 3)
    r = conv2d_grad_wrt_inputs(
            output_grad=input,
            filters=filters,
            input_shape=(None, None, input.shape[2], input.shape[3]),
            border_mode=border_mode,
            subsample=stride,
            filter_flip=True)
    if biases is None:
        return r
    else:
        return r + biases.dimshuffle('x', 0, 'x', 'x')


def t_conv_out_size(input_size, filter_size, stride, pad):
    # Author: Francesco Visin
    """Computes the length of the output of a transposed convolution
    Parameters
    ----------
    input_size : int, Iterable or Theano tensor
        The size of the input of the transposed convolution
    filter_size : int, Iterable or Theano tensor
        The size of the filter
    stride : int, Iterable or Theano tensor
        The stride of the transposed convolution
    pad : int, Iterable, Theano tensor or string
        The padding of the transposed convolution
    """
    if input_size is None:
        return None
    input_size = np.array(input_size)
    filter_size = np.array(filter_size)
    stride = np.array(stride)
    if isinstance(pad, (int, Iterable)) and not isinstance(pad, str):
        pad = np.array(pad)
        output_size = (input_size - 1) * stride + filter_size - 2*pad

    elif pad == 'full':
        output_size = input_size * stride - filter_size - stride + 2
    elif pad == 'valid':
        output_size = (input_size - 1) * stride + filter_size
    elif pad == 'same':
        output_size = input_size
    return output_size


def gru_weights(input_dim, hidden_dim, forward_init=None, hidden_init="normal",
                random_state=None):
    if random_state is None:
        raise ValueError("Must pass random_state!")
    shape = (input_dim, hidden_dim)
    if forward_init == "normal":
        W = np.hstack([np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state)])
    elif forward_init == "fan":
        W = np.hstack([np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state)])
    elif forward_init == None:
        if input_dim == hidden_dim:
            W = np.hstack([np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state)])
        else:
            # lecun
            W = np.hstack([np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state)])
    else:
        raise ValueError("Unknown forward init type %s" % forward_init)
    b = np_zeros((3 * shape[1],))

    if hidden_init == "normal":
        Wur = np.hstack([np_normal((shape[1], shape[1]), random_state),
                         np_normal((shape[1], shape[1]), random_state), ])
        U = np_normal((shape[1], shape[1]), random_state)
    elif hidden_init == "ortho":
        Wur = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                         np_ortho((shape[1], shape[1]), random_state), ])
        U = np_ortho((shape[1], shape[1]), random_state)
    return W, b, Wur, U


def GRU(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
        name=None, init=None, scale="default", weight_norm=None, biases=True):
        if init == None:
            hidden_init="ortho"
        else:
            raise ValueError("Not yet configured for other inits")
        _, _, Wur, U = gru_weights(input_dim, hidden_dim,
                                   hidden_init=hidden_init,
                                   random_state=random_state)
        dim = hidden_dim
        f1 = Linear([previous_state], [2 * hidden_dim], 2 * hidden_dim,
                    random_state, name=(name, "update/reset"), init=[Wur],
                    biases=False)
        gates = tensor.nnet.sigmoid(f1 + gate_inp)
        update = gates[:, :dim]
        reset = gates[:, dim:]
        state_reset = previous_state * reset
        f2 = Linear([state_reset], [hidden_dim], hidden_dim,
                    random_state, name=(name, "state"), init=[U], biases=False)
        next_state = tensor.tanh(f2 + inp)
        next_state = next_state * update + previous_state * (1 - update)
        return next_state


def GRUFork(list_of_inputs, input_dims, output_dim, random_state, name=None,
            init=None, scale="default", weight_norm=None, biases=True):
        gates = Linear(list_of_inputs, input_dims, 3 * output_dim,
                       random_state=random_state,
                       name=(name, "gates"), init=init, scale=scale,
                       weight_norm=weight_norm,
                       biases=biases)
        dim = output_dim
        if gates.ndim == 2:
            d = gates[:, :dim]
            g = gates[:, dim:]
        elif gates.ndim == 3:
            d = gates[:, :, :dim]
            g = gates[:, :, dim:]
        else:
            raise ValueError("Unsupported ndim")
        return d, g


def logsumexp(x, axis=None):
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max),
                              axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def softmax(X):
    # should work for both 2D and 3D
    dim = X.ndim
    e_X = tensor.exp(X - X.max(axis=dim - 1, keepdims=True))
    out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
    return out


def numpy_softmax(X, temperature=1.):
    # should work for both 2D and 3D
    dim = X.ndim
    X = X / temperature
    e_X = np.exp((X - X.max(axis=dim - 1, keepdims=True)))
    out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
    return out


def elu(x, alpha=1):
    """
    Compute the element-wise exponential linear activation function.
    From theano 0.0.8 - here for backwards compat
    """
    return tensor.switch(x > 0, x, alpha * (tensor.exp(x) - 1))


def relu(x):
    return x * (x > 1e-6)


def tanh(x):
    return tensor.tanh(x)


def sigmoid(x):
    return tensor.nnet.sigmoid(x)


def theano_one_hot(t, n_classes=None):
    if n_classes is None:
        n_classes = tensor.max(t) + 1
    ranges = tensor.shape_padleft(tensor.arange(n_classes), t.ndim)
    return tensor.eq(ranges, tensor.shape_padright(t, 1))


def binary_crossentropy(predicted_values, true_values):
    """
    Bernoulli negative log likelihood of predicted compared to binary
    true_values

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a sigmoid_layer

    true_values : tensor, shape 2D or 3D
        The ground truth values. Mush have same shape as predicted_values

    Returns
    -------
    binary_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def categorical_crossentropy(predicted_values, true_values, eps=0.):
    """
    Multinomial negative log likelihood of predicted compared to one hot
    true_values

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of a softmax

    true_values : tensor, shape 2D or 3D
        Ground truth one hot values

    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    tv = true_values.reshape((-1, true_values.shape[-1]))
    indices = tensor.argmax(tv, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if eps > 0:
        p = tensor.cast(predicted_values, theano.config.floatX) + eps
        p /= tensor.sum(p, axis=predicted_values.ndim - 1, keepdims=True)
    else:
        p = tensor.cast(predicted_values, theano.config.floatX)
    if predicted_values.ndim < 3:
        return -tensor.log(p)[rows, indices]
    elif predicted_values.ndim >= 3:
        shp = predicted_values.shape
        pred = p.reshape((-1, shp[-1]))
        ind = indices.reshape((-1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape(shp[:-1])


def sample_binomial(coeff, n_bins, theano_rng, debug=False):
    # ? Normal approximation?
    if coeff.ndim > 2:
        raise ValueError("Unsupported dim")
    if debug:
        idx = coeff * n_bins
    else:
        shp = coeff.shape
        inc = tensor.ones((shp[0], shp[1], n_bins))
        expanded_coeff = coeff.dimshuffle(0, 1, 'x')
        expanded_coeff = expanded_coeff * inc
        # n > 1 not supported?
        # idx = theano_rng.binomial(n=n_bins, p=coeff, dtype=coeff.dtype)
        idx = theano_rng.binomial(n=1, p=expanded_coeff, dtype=coeff.dtype,
                                  size=expanded_coeff.shape)
        idx = idx.sum(axis=-1)
    return tensor.cast(idx, theano.config.floatX)


def sample_softmax(coeff, theano_rng, epsilon=1E-5, debug=False):
    if coeff.ndim > 2:
        raise ValueError("Unsupported dim")
    if debug:
        idx = coeff.argmax(axis=1)
    else:
        idx = tensor.argmax(theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype),
                            axis=1)
    return tensor.cast(idx, theano.config.floatX)


def numpy_sample_softmax(coeff, random_state, debug=False):
    if coeff.ndim > 2:
        raise ValueError("Unsupported dim")
    if debug:
        idx = coeff.argmax(axis=1)
    else:
        # renormalize to avoid numpy errors about summation...
        coeff = coeff / (coeff.sum(axis=1, keepdims=True) + 1E-6)
        idxs = [np.argmax(random_state.multinomial(1, pvals=coeff[i]))
                for i in range(len(coeff))]
        idx = np.array(idxs)
    return idx.astype(theano.config.floatX)


def sample_diagonal_gmm(mu, sigma, coeff, theano_rng, epsilon=1E-5,
                        debug=False):
    if debug:
        idx = tensor.argmax(coeff, axis=1)
    else:
        idx = tensor.argmax(
            theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype), axis=1)
    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    if debug:
        z = 0.
    else:
        z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)
    s = mu + sigma * z
    return s


def sample_single_dimensional_gmms(mu, sigma, coeff, theano_rng, epsilon=1E-5,
                                   debug=False):
    if debug:
        idx = tensor.argmax(coeff, axis=1)
    else:
        idx = tensor.argmax(
            theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype), axis=1)
    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    if debug:
        z = 0.
    else:
        z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)
    s = mu + sigma * z
    return s


def diagonal_gmm(true, mu, sigma, coeff, epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr((true - mu) / sigma)
    inner = -0.5 * tensor.sum(inner, axis=1)
    nll = -logsumexp(tensor.log(coeff) + inner, axis=1)
    nll = nll.reshape(shape_t[:-1], ndim=n_dim-1)
    return nll


def diagonal_phase_gmm(true, mu, sigma, coeff, epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner0 = np.pi - abs(tensor.mod(true - mu, 2 * np.pi) - np.pi)
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr(inner0 / sigma)
    inner = -0.5 * tensor.sum(inner, axis=1)
    nll = -logsumexp(tensor.log(coeff) + inner, axis=1)
    nll = nll.reshape(shape_t[:-1], ndim=n_dim-1)
    return nll


def single_dimensional_gmms(true, mu, sigma, coeff, epsilon=1E-5):
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr((true - mu) / sigma)
    inner = -0.5 * inner
    nll = -logsumexp(tensor.sum(tensor.log(coeff) + inner, axis=1), axis=1)
    nll = nll.reshape((shape_t[0], shape_t[1]))
    return nll


def single_dimensional_phase_gmms(true, mu, sigma, coeff, epsilon=1E-5):
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')
    inner0 = np.pi - abs(tensor.mod(true - mu, 2 * np.pi) - np.pi)
    inner = tensor.log(2 * np.pi) + 2 * tensor.log(sigma)
    inner += tensor.sqr(inner0 / sigma)
    inner = -0.5 * inner
    nll = -logsumexp(tensor.sum(tensor.log(coeff) + inner, axis=1), axis=1)
    nll = nll.reshape((shape_t[0], shape_t[1]))
    return nll


def bernoulli_and_bivariate_gmm(true, mu, sigma, corr, coeff, binary,
                                epsilon=1E-5):
    n_dim = true.ndim
    shape_t = true.shape
    true = true.reshape((-1, shape_t[-1]))
    true = true.dimshuffle(0, 1, 'x')

    mu_1 = mu[:, 0, :]
    mu_2 = mu[:, 1, :]

    sigma_1 = sigma[:, 0, :]
    sigma_2 = sigma[:, 1, :]

    binary = (binary + epsilon) * (1 - 2 * epsilon)

    c_b = tensor.sum(tensor.xlogx.xlogy0(true[:, 0],  binary) + tensor.xlogx.xlogy0(
        1 - true[:, 0], 1 - binary), axis=1)

    inner1 = (0.5 * tensor.log(1. - corr ** 2 + epsilon))
    inner1 += tensor.log(sigma_1) + tensor.log(sigma_2)
    inner1 += tensor.log(2. * np.pi)

    t1 = true[:, 1]
    t2 = true[:, 2]
    Z = (((t1 - mu_1)/sigma_1)**2) + (((t2 - mu_2) / sigma_2)**2)
    Z -= (2. * (corr * (t1 - mu_1)*(t2 - mu_2)) / (sigma_1 * sigma_2))
    inner2 = 0.5 * (1. / (1. - corr**2 + epsilon))
    cost = - (inner1 + (inner2 * Z))

    nll = -logsumexp(tensor.log(coeff) + cost, axis=1)
    nll -= c_b
    return nll.reshape(shape_t[:-1], ndim=n_dim-1)


def sample_bernoulli_and_bivariate_gmm(mu, sigma, corr, coeff, binary,
                                       theano_rng, epsilon=1E-5):

    idx = tensor.argmax(theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype),
                        axis=1)

    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
    corr = corr[tensor.arange(corr.shape[0]), idx]

    mu_x = mu[:, 0]
    mu_y = mu[:, 1]
    sigma_x = sigma[:, 0]
    sigma_y = sigma[:, 1]

    z = theano_rng.normal(size=mu.shape, avg=0., std=1., dtype=mu.dtype)

    un = theano_rng.uniform(size=binary.shape)
    binary = tensor.cast(un < binary, theano.config.floatX)

    s_x = (mu_x + sigma_x * z[:, 0]).dimshuffle(0, 'x')
    s_y = mu_y + sigma_y * (
        (z[:, 0] * corr) + (z[:, 1] * tensor.sqrt(1. - corr ** 2)))
    s_y = s_y.dimshuffle(0, 'x')
    s = tensor.concatenate([binary, s_x, s_y], axis=1)
    return s
"""
end initializers and Theano functions
"""

"""
start optimizers
"""
def gradient_clipping(grads, rescale=5.):
    grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
    scaling_num = rescale
    scaling_den = tensor.maximum(rescale, grad_norm)
    scaling = scaling_num / scaling_den
    return [g * scaling for g in grads]


class sgd(object):
    """
    Vanilla SGD
    """
    def __init__(self, params, learning_rate):
        self.learning_rate = as_shared(learning_rate)

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates


class sgd_momentum(object):
    """
    SGD with momentum
    """
    def __init__(self, params, learning_rate, momentum):
        self.learning_rate = as_shared(learning_rate)
        self.momentum = momentum
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        momentum = self.momentum
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            updates.append((param, param - learning_rate * grad))
            updates.append((memory, momentum * memory + (1. - momentum) * grad))
        return updates


class sgd_nesterov(object):
    """
    SGD with nesterov momentum

    Based on example from Yann D.

    See Formula 7 from
    Advances in Optimizing Recurrent Neural Networks
    Y. Benio, N. Boulanger-Lewandowski, R. Pascanu
    """
    def __init__(self, params, learning_rate, momentum):
        self.learning_rate = as_shared(learning_rate)
        self.momentum = momentum
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        momentum = self.momentum
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params, learning_rate, momentum, rescale=5.):
        self.learning_rate = as_shared(learning_rate)
        self.momentum = momentum
        self.rescale = rescale
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.squared_memory_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.momentum_memory_ = [theano.shared(np.zeros_like(p.get_value()))
                                 for p in params]

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        momentum = self.momentum
        rescale = self.rescale
        grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
        scaling_num = rescale
        scaling_den = tensor.maximum(rescale, grad_norm)
        scaling = scaling_num / scaling_den
        # constants, from AG "Generating Sequences with Recurrent Neural
        # Networks"
        decay = 0.95
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad *= scaling
            memory = self.memory_[n]
            squared_memory = self.squared_memory_[n]
            momentum_memory = self.momentum_memory_[n]
            grad_gi = decay * memory + (1 - decay) * grad
            decayed_ni = decay * squared_memory + (1 - decay) * grad ** 2
            grad_scaled = grad / tensor.sqrt(
                decayed_ni - grad_gi ** 2 + minimum_grad)
            update = momentum * momentum_memory - learning_rate * grad_scaled
            update2 = momentum * momentum * momentum_memory - (
                1 + momentum) * learning_rate * grad_scaled
            updates.append((memory, grad_gi))
            updates.append((squared_memory, decayed_ni))
            updates.append((momentum_memory, update))
            updates.append((param, param + update2))
        return updates


class adagrad(object):
    """
    Adagrad optimizer
    """
    def __init__(self, params, learning_rate, eps=1E-8):
        self.learning_rate = as_shared(learning_rate)
        self.eps = eps
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        eps = self.eps
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            m_t = memory + grad ** 2
            g_t = grad / (eps + tensor.sqrt(m_t))
            p_t = param - learning_rate * g_t
            updates.append((memory, m_t))
            updates.append((param, p_t))
        return updates


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


class adam(object):
    """
    Adam optimizer

    Based on implementation from @NewMu / Alex Radford
    """
    def __init__(self, params, learning_rate, b1=0.1, b2=0.001, eps=1E-8):
        self.learning_rate = as_shared(learning_rate)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.velocity_ = [theano.shared(np.zeros_like(p.get_value()))
                          for p in params]
        self.itr_ = theano.shared(np.array(0.).astype(theano.config.floatX))

    def updates(self, params, grads):
        learning_rate = self.learning_rate
        b1 = self.b1
        b2 = self.b2
        eps = self.eps
        updates = []
        itr = self.itr_
        i_t = itr + 1.
        fix1 = 1. - (1. - b1) ** i_t
        fix2 = 1. - (1. - b2) ** i_t
        lr_t = learning_rate * (tensor.sqrt(fix2) / fix1)
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            velocity = self.velocity_[n]
            m_t = (b1 * grad) + ((1. - b1) * memory)
            v_t = (b2 * tensor.sqr(grad)) + ((1. - b2) * velocity)
            g_t = m_t / (tensor.sqrt(v_t) + eps)
            p_t = param - (lr_t * g_t)
            updates.append((memory, m_t))
            updates.append((velocity, v_t))
            updates.append((param, p_t))
        updates.append((itr, i_t))
        return updates
"""
end optimizers
"""

"""
start training utilities
"""
def get_shared_variables_from_function(func):
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    return shared_variables


def get_values_from_function(func):
    return [v.get_value() for v in get_shared_variables_from_function(func)]


def safe_zip(*args):
    """Like zip, but ensures arguments are of same length.

       Borrowed from pylearn2
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def set_shared_variables_in_function(func, list_of_values):
    # TODO : Add checking that sizes are OK
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    [s.set_value(v) for s, v in safe_zip(shared_variables, list_of_values)]


def save_weights(save_path, items_dict, use_resource_dir=True):
    logger.info("Not saving weights due to copy issues in npz")
    return
    weights_dict = {}
    # k is the function name, v is a theano function
    for k, v in items_dict.items():
        if isinstance(v, theano.compile.function_module.Function):
            # w is all the numpy values from a function
            w = get_values_from_function(v)
            for n, w_v in enumerate(w):
                weights_dict[k + "_%i" % n] = w_v
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    logger.info("Saving weights to %s" % save_weights_path)
    if len(weights_dict.keys()) > 0:
        np.savez(save_path, **weights_dict)
    else:
        logger.info("Possible BUG: no theano functions found in items_dict, "
              "unable to save weights!")
    logger.info("Weight saving complete %s" % save_path)


@coroutine
def threaded_weights_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, checkpoint_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, items_dict = item
                save_weights(save_path, items_dict)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


def save_checkpoint(save_path, pickle_item, use_resource_dir=True):
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    sys.setrecursionlimit(40000)
    logger.info("Saving checkpoint to %s" % save_path)
    with open(save_path, mode="wb") as f:
        pickle.dump(pickle_item, f, protocol=-1)
    logger.info("Checkpoint saving complete %s" % save_path)


@coroutine
def threaded_checkpoint_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, checkpoint_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, pickle_item = item
                save_checkpoint(save_path, pickle_item)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


def load_checkpoint(saved_checkpoint_path):
    logger.info("Loading checkpoint from %s" % saved_checkpoint_path)
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(saved_checkpoint_path, mode="rb") as f:
        pickle_item = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return pickle_item


def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    partial_path = get_resource_dir("js_plot_dependencies")
    full_path = os.path.join(partial_path, "master.zip")
    url = "http://github.com/kastnerkyle/simple_template_plotter/archive/master.zip"
    if not os.path.exists(full_path):
        logger.info("Downloading plotter template code from %s" % url)
        download(url, full_path)
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(partial_path)
        zip_ref.close()

    js_path = os.path.join(partial_path, "simple_template_plotter-master")
    template_path =  os.path.join(js_path, "template.html")
    f = open(template_path, mode='r')
    all_template_lines = f.readlines()
    f.close()
    imports_split_index = [n for n, l in enumerate(all_template_lines)
                           if "IMPORTS_SPLIT" in l][0]
    data_split_index = [n for n, l in enumerate(all_template_lines)
                        if "DATA_SPLIT" in l][0]
    log_split_index = [n for n, l in enumerate(all_template_lines)
                       if "LOGGING_SPLIT" in l][0]
    first_part = all_template_lines[:imports_split_index]
    imports_part = []
    js_files_path = os.path.join(js_path, "js")
    js_file_names = ["jquery-1.9.1.js", "knockout-3.0.0.js",
                     "highcharts.js", "exporting.js"]
    js_files = [os.path.join(js_files_path, jsf) for jsf in js_file_names]
    for js_file in js_files:
        with open(js_file, "r") as f:
            imports_part.extend(
                ["<script>\n"] + f.readlines() + ["</script>\n"])
    post_imports_part = all_template_lines[
        imports_split_index + 1:data_split_index]
    log_part = all_template_lines[data_split_index + 1:log_split_index]
    last_part = all_template_lines[log_split_index + 1:]

    def gen_js_field_for_key_value(key, values, show=True):
        assert type(values) is list
        if isinstance(values[0], (np.generic, np.ndarray)):
            values = [float(v.ravel()) for v in values]
        maxlen = 1500
        if len(values) > maxlen:
            values = list(np.interp(np.linspace(0, len(values), maxlen),
                          np.arange(len(values)), values))
        show_key = "true" if show else "false"
        return "{\n    name: '%s',\n    data: %s,\n    visible: %s\n},\n" % (
            str(key), str(values), show_key)
    data_part = [gen_js_field_for_key_value(k, results_dict[k], True)
                 if k in default_show or default_show == "all"
                 else gen_js_field_for_key_value(k, results_dict[k], False)
                 for k in sorted(results_dict.keys())]
    all_filled_lines = first_part + imports_part + post_imports_part
    all_filled_lines = all_filled_lines + data_part + log_part
    # add logging output
    tmp = copy.copy(string_f)
    tmp.seek(0)
    log_output = tmp.readlines()
    del tmp
    all_filled_lines = all_filled_lines + log_output + last_part
    return all_filled_lines


def save_results_as_html(save_path, results_dict, use_resource_dir=True,
                         default_no_show="_auto"):
    show_keys = [k for k in results_dict.keys()
                 if default_no_show not in k]
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=show_keys)
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    logger.info("Saving HTML results %s" % save_path)
    with open(save_path, "w") as f:
        f.writelines(as_html)
    logger.info("Completed HTML results saving %s" % save_path)


@coroutine
def threaded_html_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, results_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, results_dict = item
                save_results_as_html(save_path, results_dict)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


def implot(arr, title="", cmap="gray", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.matshow(arr, cmap=cmap)
    plt.axis("off")

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

    x1 = arr.shape[0]
    y1 = arr.shape[1]
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)

def _archive(tag=None):
    script_name = get_script_name()[:-3]
    save_path = get_resource_dir(script_name)
    if tag is None:
        save_script_path = os.path.join(save_path, get_script_name())
    else:
        save_script_path = os.path.join(save_path, tag + "_" + get_script_name())

    logger.info("Saving code archive for %s" % (save_path))
    script_location = os.path.abspath(sys.argv[0])
    shutil.copy2(script_location, save_script_path)

    lib_location = os.path.realpath(__file__)
    lib_name = lib_location.split(os.sep)[-1]
    if tag is None:
        save_lib_path = os.path.join(save_path, lib_name)
    else:
        save_lib_path = os.path.join(save_path, tag + "_" + lib_name)
    shutil.copy2(lib_location, save_lib_path)


class Igor(object):
    """
    Runs the loop - thin wrapper for serializing

    Named in reference to https://www.reddit.com/r/MachineLearning/comments/4kd1qp/good_code_to_learn_from/d3e2irr
    """
    def __init__(self, loop_function, train_function, train_itr,
                 valid_function, valid_itr, n_epochs, checkpoint_dict,
                 checkpoint_delay=10,
                 checkpoint_every_n_epochs=1,
                 checkpoint_every_n_updates=np.inf,
                 checkpoint_every_n_seconds=np.inf,
                 monitor_frequency=1000,
                 skip_minimums=False,
                 skip_intermediates=True,
                 skip_most_recents=False):
        self.loop_function = loop_function
        self.train_function = train_function
        self.train_itr = train_itr
        self.valid_function = valid_function
        self.valid_itr = valid_itr
        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

        # These parameters should be serialized
        self.checkpoint_delay = checkpoint_delay
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_updates = checkpoint_every_n_updates
        self.checkpoint_every_n_seconds = checkpoint_every_n_seconds
        self.monitor_frequency = monitor_frequency
        self.skip_minimums = skip_minimums
        self.skip_intermediates = skip_intermediates
        self.skip_most_recents = skip_most_recents

        # tracker to ensure restarting at the correct minibatch
        self.num_train_minibatches_run = -1

    def __getstate__(self):
        skiplist = [self.loop_function,
                    self.train_function,
                    self.train_itr,
                    self.valid_function,
                    self.valid_itr,
                    self.n_epochs,
                    self.checkpoint_dict]
        return {k:v for k, v in self.__dict__.items() if v not in skiplist}

    def refresh(self, loop_function, train_function, train_itr,
                valid_function, valid_itr, n_epochs,
                checkpoint_dict):
        # Must refresh after reloading from pkl
        self.loop_function = loop_function
        self.train_function = train_function
        self.train_itr = train_itr
        self.valid_function = valid_function
        self.valid_itr = valid_itr
        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

    def run(self):
        run_loop(self.loop_function, self.train_function, self.train_itr,
                 self.valid_function, self.valid_itr, self.n_epochs,
                 self.checkpoint_dict,
                 self.checkpoint_delay,
                 self.checkpoint_every_n_epochs,
                 self.checkpoint_every_n_updates,
                 self.checkpoint_every_n_seconds,
                 self.monitor_frequency,
                 self.skip_minimums,
                 self.skip_intermediates,
                 self.skip_most_recents,
                 self.num_train_minibatches_run,
                 self)

def run_loop(loop_function, train_function, train_itr,
             valid_function, valid_itr, n_epochs, checkpoint_dict,
             checkpoint_delay=10, checkpoint_every_n_epochs=1,
             checkpoint_every_n_updates=np.inf,
             checkpoint_every_n_seconds=np.inf,
             monitor_frequency=1000, skip_minimums=False,
             skip_intermediates=True, skip_most_recents=False,
             skip_n_train_minibatches=-1,
             stateful_object=None):
    """
    TODO: add all logging info into the js report
    TODO: add upload fields to add data to an html and save a copy
    loop function should return a list of costs
    stateful_object allows to serialize and relaunch in middle of an epoch
    for long training models
    """
    logger.info("Running loop...")
    # Assume keys which are theano functions to ignore!
    ignore_keys = [k for k, v in checkpoint_dict.items()
                   if isinstance(v, theano.compile.function_module.Function)]

    _loop = loop_function
    ident = str(uuid.uuid4())[:8]
    random_state = np.random.RandomState(2177)
    monitor_prob = 1. / monitor_frequency

    non_ignored_keys = [k for k in checkpoint_dict.keys()
                        if k not in ignore_keys]
    if len(non_ignored_keys) > 0:
        overall_train_costs = checkpoint_dict["train_costs"]
        overall_valid_costs = checkpoint_dict["valid_costs"]
        # Auto tracking times
        overall_epoch_deltas = checkpoint_dict["epoch_deltas_auto"]
        overall_epoch_times = checkpoint_dict["epoch_times_auto"]
        overall_train_deltas = checkpoint_dict["train_deltas_auto"]
        overall_train_times = checkpoint_dict["train_times_auto"]
        overall_valid_deltas = checkpoint_dict["valid_deltas_auto"]
        overall_valid_times = checkpoint_dict["valid_times_auto"]
        overall_checkpoint_deltas = checkpoint_dict["checkpoint_deltas_auto"]
        overall_checkpoint_times = checkpoint_dict["checkpoint_times_auto"]
        overall_joint_deltas = checkpoint_dict["joint_deltas_auto"]
        overall_joint_times = checkpoint_dict["joint_times_auto"]
        overall_train_checkpoint = checkpoint_dict["train_checkpoint_auto"]
        overall_valid_checkpoint = checkpoint_dict["valid_checkpoint_auto"]
        keys_checked = ["train_costs",
                        "valid_costs",
                        "epoch_deltas_auto",
                        "epoch_times_auto",
                        "train_deltas_auto",
                        "train_times_auto",
                        "valid_deltas_auto",
                        "valid_times_auto",
                        "checkpoint_deltas_auto",
                        "checkpoint_times_auto",
                        "joint_deltas_auto",
                        "joint_times_auto",
                        "train_checkpoint_auto",
                        "valid_checkpoint_auto"]
        not_handled = [k for k in checkpoint_dict.keys()
                       if k not in keys_checked and k not in ignore_keys]
        if len(not_handled) > 0:
            raise ValueError("Unhandled keys %s in checkpoint_dict, exiting..." % not_handled)

        epoch_time_total = overall_epoch_times[-1]
        train_time_total = overall_train_times[-1]
        valid_time_total = overall_valid_times[-1]
        checkpoint_time_total = overall_checkpoint_times[-1]
        joint_time_total = overall_joint_times[-1]

        start_epoch = len(overall_train_costs)
    else:
        overall_train_costs = []
        overall_valid_costs = []
        overall_train_checkpoint = []
        overall_valid_checkpoint = []

        epoch_time_total = 0
        train_time_total = 0
        valid_time_total = 0
        checkpoint_time_total = 0
        joint_time_total = 0
        overall_epoch_times = []
        overall_epoch_deltas = []
        overall_train_times = []
        overall_train_deltas = []
        overall_valid_times = []
        overall_valid_deltas = []
        # Add zeros to avoid errors
        overall_checkpoint_times = [0]
        overall_checkpoint_deltas = [0]
        overall_joint_times = [0]
        overall_joint_deltas = [0]

        start_epoch = 0

    # save current state of kdllib and calling script
    _archive(ident)

    tcw = threaded_checkpoint_writer()
    tww = threaded_weights_writer()
    thw = threaded_html_writer()

    best_train_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    best_valid_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    # If there are more than 1M minibatches per epoch this will break!
    # Not reallocating buffer greatly helps fast training models though
    # Also we have bigger problems if there are 1M minibatches per epoch...
    # This will get sliced down to the correct number of minibatches down below
    train_costs = [0.] * 1000000
    valid_costs = [0.] * 1000000
    try:
        for e in range(start_epoch, start_epoch + n_epochs):
            joint_start = time.time()
            epoch_start = time.time()
            logger.info(" ")
            logger.info("Starting training, epoch %i" % e)
            logger.info(" ")
            train_mb_count = 0
            valid_mb_count = 0
            results_dict = {k: v for k, v in checkpoint_dict.items()
                            if k not in ignore_keys}
            this_results_dict = results_dict
            try:
                train_start = time.time()
                last_time_checkpoint = train_start
                while True:
                    if train_mb_count < skip_n_train_minibatches:
                        train_mb_count += 1
                        continue
                    partial_train_costs = _loop(train_function, train_itr)
                    train_costs[train_mb_count] = np.mean(partial_train_costs)
                    tc = train_costs[train_mb_count]
                    if np.isnan(tc):
                        logger.info("NaN detected in train cost, update %i" % train_mb_count)
                        raise StopIteration("NaN detected in train")

                    train_mb_count += 1
                    if (train_mb_count % checkpoint_every_n_updates) == 0:
                        checkpoint_save_path = "%s_model_update_checkpoint_%i.pkl" % (ident, train_mb_count)
                        weights_save_path = "%s_model_update_weights_%i.npz" % (ident, train_mb_count)
                        results_save_path = "%s_model_update_results_%i.html" % (ident, train_mb_count)
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        copy_pickle = pickle.dumps(checkpoint_dict)
                        copy_dict = pickle.loads(copy_pickle)
                        tcw.send((checkpoint_save_path, copy_dict))
                        tww.send((weights_save_path, copy_dict))
                        logger.info(" ")
                        logger.info("Update checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean
                        thw.send((results_save_path, this_results_dict))
                        if stateful_object is not None:
                            stateful_object.num_train_minibatches_run = train_mb_count
                            object_save_path = "%s_model_update_object_%i.pkl" % (ident, train_mb_count)
                            save_checkpoint(object_save_path, stateful_object)
                    elif (time.time() - last_time_checkpoint) >= checkpoint_every_n_seconds:
                        time_diff = time.time() - train_start
                        last_time_checkpoint = time.time()
                        checkpoint_save_path = "%s_model_time_checkpoint_%i.pkl" % (ident, int(time_diff))
                        weights_save_path = "%s_model_time_weights_%i.npz" % (ident, int(time_diff))
                        results_save_path = "%s_model_time_results_%i.html" % (ident, int(time_diff))
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        copy_pickle = pickle.dumps(checkpoint_dict)
                        copy_dict = pickle.loads(copy_pickle)
                        tcw.send((checkpoint_save_path, copy_dict))
                        tww.send((weights_save_path, copy_dict))
                        logger.info(" ")
                        logger.info("Time checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean
                        thw.send((results_save_path, this_results_dict))
                        if stateful_object is not None:
                            stateful_object.num_train_minibatches_run = train_mb_count
                            object_save_path = "%s_model_time_object_%i.pkl" % (ident, int(time_diff))
                            save_checkpoint(object_save_path, stateful_object)
                    draw = random_state.rand()
                    if draw < monitor_prob and not skip_intermediates:
                        logger.info(" ")
                        logger.info("Starting train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        results_save_path = "%s_intermediate_results.html" % ident
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        thw.send((results_save_path, this_results_dict))
            except StopIteration:
                # Slice so that only valid data is in the minibatch
                # this also assumes there is not a variable number
                # of minibatches in an epoch!
                train_stop = time.time()
                train_costs = train_costs[:train_mb_count]
                logger.info(" ")
                logger.info("Starting validation, epoch %i" % e)
                logger.info(" ")
                valid_start = time.time()
                try:
                    while True:
                        partial_valid_costs = _loop(valid_function, valid_itr)
                        valid_costs[valid_mb_count] = np.mean(partial_valid_costs)
                        vc = valid_costs[valid_mb_count]
                        if np.isnan(vc):
                            logger.info("NaN detected in valid cost, minibatch %i" % valid_mb_count)
                            raise StopIteration("NaN detected in valid")
                        valid_mb_count += 1
                        draw = random_state.rand()
                        if draw < monitor_prob and not skip_intermediates:
                            logger.info(" ")
                            logger.info("Valid mb %i" % valid_mb_count)
                            logger.info("Current validation mean cost %f" % np.mean(
                                valid_costs))
                            logger.info(" ")
                            results_save_path = "%s_intermediate_results.html" % ident
                            this_results_dict["this_epoch_valid_auto"] = valid_costs[:valid_mb_count]
                            thw.send((results_save_path, this_results_dict))
                except StopIteration:
                    pass
                logger.info(" ")
                valid_stop = time.time()
                epoch_stop = time.time()
                valid_costs = valid_costs[:valid_mb_count]

                # Logging and tracking training statistics
                epoch_time_delta = epoch_stop - epoch_start
                epoch_time_total += epoch_time_delta
                overall_epoch_deltas.append(epoch_time_delta)
                overall_epoch_times.append(epoch_time_total)

                train_time_delta = train_stop - train_start
                train_time_total += train_time_delta
                overall_train_deltas.append(train_time_delta)
                overall_train_times.append(train_time_total)

                valid_time_delta = valid_stop - valid_start
                valid_time_total += valid_time_delta
                overall_valid_deltas.append(valid_time_delta)
                overall_valid_times.append(valid_time_total)

                mean_epoch_train_cost = np.mean(train_costs)
                # np.inf trick to avoid taking the min of length 0 list
                old_min_train_cost = min(overall_train_costs + [np.inf])
                if np.isnan(mean_epoch_train_cost):
                    logger.info("Previous train costs %s" % overall_train_costs[-5:])
                    logger.info("NaN detected in train cost, epoch %i" % e)
                    raise StopIteration("NaN detected in train")
                overall_train_costs.append(mean_epoch_train_cost)

                mean_epoch_valid_cost = np.mean(valid_costs)
                old_min_valid_cost = min(overall_valid_costs + [np.inf])
                if np.isnan(mean_epoch_valid_cost):
                    logger.info("Previous valid costs %s" % overall_valid_costs[-5:])
                    logger.info("NaN detected in valid cost, epoch %i" % e)
                    raise StopIteration("NaN detected in valid")
                overall_valid_costs.append(mean_epoch_valid_cost)

                if mean_epoch_train_cost < old_min_train_cost:
                    overall_train_checkpoint.append(mean_epoch_train_cost)
                else:
                    overall_train_checkpoint.append(old_min_train_cost)

                if mean_epoch_valid_cost < old_min_valid_cost:
                    overall_valid_checkpoint.append(mean_epoch_valid_cost)
                else:
                    overall_valid_checkpoint.append(old_min_valid_cost)

                checkpoint_dict["train_costs"] = overall_train_costs
                checkpoint_dict["valid_costs"] = overall_valid_costs
                # Auto tracking times
                checkpoint_dict["epoch_deltas_auto"] = overall_epoch_deltas
                checkpoint_dict["epoch_times_auto"] = overall_epoch_times
                checkpoint_dict["train_deltas_auto"] = overall_train_deltas
                checkpoint_dict["train_times_auto"] = overall_train_times
                checkpoint_dict["valid_deltas_auto"] = overall_valid_deltas
                checkpoint_dict["valid_times_auto"] = overall_valid_times
                checkpoint_dict["checkpoint_deltas_auto"] = overall_checkpoint_deltas
                checkpoint_dict["checkpoint_times_auto"] = overall_checkpoint_times
                checkpoint_dict["joint_deltas_auto"] = overall_joint_deltas
                checkpoint_dict["joint_times_auto"] = overall_joint_times
                # Tracking if checkpoints are made
                checkpoint_dict["train_checkpoint_auto"] = overall_train_checkpoint
                checkpoint_dict["valid_checkpoint_auto"] = overall_valid_checkpoint

                script = get_script_name()
                hostname = socket.gethostname()
                logger.info("Host %s, script %s" % (hostname, script))
                logger.info("Epoch %i complete" % e)
                logger.info("Epoch mean train cost %f" % mean_epoch_train_cost)
                logger.info("Epoch mean valid cost %f" % mean_epoch_valid_cost)
                logger.info("Previous train costs %s" % overall_train_costs[-5:])
                logger.info("Previous valid costs %s" % overall_valid_costs[-5:])

                results_dict = {k: v for k, v in checkpoint_dict.items()
                                if k not in ignore_keys}

                # Checkpointing part
                checkpoint_start = time.time()
                if e < checkpoint_delay or skip_minimums:
                    pass
                elif mean_epoch_valid_cost < old_min_valid_cost:
                    logger.info("Checkpointing valid...")
                    # Using dumps so relationship between keys in the pickle
                    # is preserved
                    best_valid_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                    best_valid_checkpoint_epoch = e
                    if mean_epoch_train_cost < old_min_train_cost:
                        best_train_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                        best_train_checkpoint_epoch = e
                    logger.info("Valid checkpointing complete.")
                elif mean_epoch_train_cost < old_min_train_cost:
                    logger.info("Checkpointing train...")
                    best_train_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                    best_train_checkpoint_epoch = e
                    logger.info("Train checkpointing complete.")

                if e < checkpoint_delay:
                    pass
                    # Don't skip force checkpoints after default delay
                    # Printing already happens above
                elif((e % checkpoint_every_n_epochs) == 0) or (e == (n_epochs - 1)):
                    logger.info("Checkpointing force...")
                    checkpoint_save_path = "%s_model_checkpoint_%i.pkl" % (ident, e)
                    weights_save_path = "%s_model_weights_%i.npz" % (ident, e)
                    results_save_path = "%s_model_results_%i.html" % (ident, e)
                    # Use pickle to preserve relationships between keys
                    # while still copying buffers
                    copy_pickle = pickle.dumps(checkpoint_dict)
                    copy_dict = pickle.loads(copy_pickle)
                    tcw.send((checkpoint_save_path, copy_dict))
                    tww.send((weights_save_path, copy_dict))
                    thw.send((results_save_path, results_dict))
                    logger.info("Force checkpointing complete.")

                checkpoint_stop = time.time()
                joint_stop = time.time()

                if skip_most_recents:
                    pass
                else:
                    # Save latest
                    results_save_path = "%s_most_recent_results.html" % ident
                    thw.send((results_save_path, results_dict))

                # Will show up next go around
                checkpoint_time_delta = checkpoint_stop - checkpoint_start
                checkpoint_time_total += checkpoint_time_delta
                overall_checkpoint_deltas.append(checkpoint_time_delta)
                overall_checkpoint_times.append(checkpoint_time_total)

                joint_time_delta = joint_stop - joint_start
                joint_time_total += joint_time_delta
                overall_joint_deltas.append(joint_time_delta)
                overall_joint_times.append(joint_time_total)
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user! Saving current best results.")

    if not skip_minimums:
        # Finalize saving best train and valid
        best_valid_checkpoint_dict = pickle.loads(best_valid_checkpoint_pickle)
        best_valid_results_dict = {k: v for k, v in best_valid_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_valid_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, ee)
        weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, ee)
        results_save_path = "%s_model_results_valid_%i.html" % (ident, ee)
        tcw.send((checkpoint_save_path, best_valid_checkpoint_dict))
        tww.send((weights_save_path, best_valid_checkpoint_dict))
        thw.send((results_save_path, best_valid_results_dict))

        best_train_checkpoint_dict = pickle.loads(best_train_checkpoint_pickle)
        best_train_results_dict = {k: v for k, v in best_train_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_train_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, ee)
        weights_save_path = "%s_model_weights_train_%i.npz" % (ident, ee)
        results_save_path = "%s_model_results_train_%i.html" % (ident, ee)
        tcw.send((checkpoint_save_path, best_train_checkpoint_dict))
        tww.send((weights_save_path, best_train_checkpoint_dict))
        thw.send((results_save_path, best_train_results_dict))
    logger.info("Loop finished, closing write threads (this may take a while!)")
    tcw.close()
    tww.close()
    thw.close()
"""
end training utilities
"""

"""
start examples
"""
def run_lpc_example():
    fs, X = fetch_sample_speech_tapestry()
    window_size = 256
    window_step = 128
    a, g, e = lpc_analysis(X, order=8, window_step=window_step,
                           window_size=window_size, emphasis=0.9,
                           copy=True)
    v, p = voiced_unvoiced(X, window_size=window_size,
                           window_step=window_step)
    X_r = lpc_synthesis(a, g, e, voiced_frames=v,
                        emphasis=0.9, window_step=window_step)
    wavfile.write("lpc_orig.wav", fs, soundsc(X))
    wavfile.write("lpc_rec.wav", fs, soundsc(X_r))


def run_fft_dct_example():
    # This is an example of the preproc we want to do with a lot of added noise
    random_state = np.random.RandomState(1999)

    fs, d, _ = fetch_sample_speech_fruit()
    n_fft = 128
    X = d[0]
    X_stft = stft(X, n_fft)
    X_rr = complex_to_real_view(X_stft).ravel()
    X_dct = mdct_slow(X_rr, n_fft)
    """
    X_dct_sub = X_dct[1:] - X_dct[:-1]
    std = X_dct_sub.std(axis=0, keepdims=True)
    X_dct_sub += .15 * std * random_state.randn(
        X_dct_sub.shape[0], X_dct_sub.shape[1])
    X_dct_unsub = np.cumsum(X_dct_sub, axis=0)
    X_idct = imdct_slow(X_dct_unsub, n_fft).reshape(-1, n_fft)
    """
    #std = X_dct.std(axis=0, keepdims=True)
    #X_dct[:, 80:] = 0.
    #X_dct += .8 * std * random_state.randn(
    #    X_dct.shape[0], X_dct.shape[1])
    X_idct = imdct_slow(X_dct, n_fft).reshape(-1, n_fft)
    X_irr = real_to_complex_view(X_idct)
    X_r = istft(X_irr, n_fft)[:len(X)]
    X_r = X_r - X_r.mean()

    SNR = 20 * np.log10(np.linalg.norm(X - X_r) / np.linalg.norm(X))
    print(SNR)

    wavfile.write("fftdct_orig.wav", fs, soundsc(X))
    wavfile.write("fftdct_rec.wav", fs, soundsc(X_r))


def run_blizzard_example():
    bliz = Blizzard_dataset()
    start = time.time()
    itr = 1
    while True:
        r = bliz.next()
        stop = time.time()
        tot = stop - start
        print("Threaded time: %s" % (tot))
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        itr += 1
        break
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    raise ValueError()
"""
end examples
"""

if __name__ == "__main__":
    #run_fft_dct_example()
    #run_lpc_example()
    #run_blizzard_example()
    #fetch_ono()
    fetch_walla()
