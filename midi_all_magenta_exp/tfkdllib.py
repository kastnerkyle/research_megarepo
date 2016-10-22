# License: BSD 3-clause
# Authors: Kyle Kastner
from __future__ import print_function
import numpy as np
import uuid
from scipy import linalg
import tensorflow as tf
import shutil
import socket
import wave
import os
import glob
import re
import copy
import time
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib
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
# Automatically put the HTML break characters on there for html logger
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
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.next()
        return cr
    return start
"""
end decorators
"""

"""
begin metautils
"""


def shape(x):
    # Get shape of Variable through hacky hacky string parsing
    shape_tup = repr(tf.Print(x, [tf.shape(x)])).split("shape=")[1]
    shape_tup = shape_tup.split("dtype=")[0][:-1]
    shape_tup = shape_tup.split(",")
    shape_tup = [re.sub('[^0-9?]', '', s) for s in shape_tup]
    # remove empty '' dims in 1D case
    shape_tup = tuple([int(s) if s != "?" else -1
                       for s in shape_tup if len(s) >= 1])
    if sum([1 for s in shape_tup if s < 1]) > 1:
        raise ValueError("too many ? dims")
    return shape_tup


def ndim(x):
    return len(shape(x))


def print_network(params):
    n_params = sum([np.prod(shape(p)) for p in params])
    logger.info("Total number of parameters: %fM" % (n_params / float(1E6)))


def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    # rediculously hacky string parsing... wowie
    a_tup = shape(a)
    b_tup = shape(b)
    a_i = tf.reshape(a, [-1, a_tup[-1]])
    a_n = tf.matmul(a_i, b)
    a_n = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
    return a_n


def ni_slice(sub_values, last_ind, axis=0):
    # TODO: Allow both to be negative indexed...
    ndim = len(shape(sub_values))
    im1 = 0 + abs(last_ind)
    i = [[None, None]] * ndim
    i[axis] = [im1, None]
    am = [False] * ndim
    am[axis] = True
    sl = [slice(*ii) for ii in i]
    ti = tf.reverse(sub_values, am)[sl]
    return tf.reverse(ti, am)


def ni(t, ind, axis=0):
    # Negative single index helper
    ndim = len(shape(t))
    im1 = -1 + abs(ind)
    i = [[None, None]] * ndim
    i[axis] = [im1, im1 + 1]
    am = [False] * ndim
    am[axis] = True
    sl = [slice(*ii) for ii in i]
    ti = tf.reverse(t, am)[sl]
    return ti[0, :, :]


def scan(fn, sequences, outputs_info):
    # for some reason TF step needs initializer passed as first argument?
    # a tiny wrapper to tf.scan to make my life easier
    # closer to theano scan, allows for step functions with multiple arguments
    # may eventually have kwargs which match theano
    for i in range(len(sequences)):
        # Try to accomodate for masks...
        seq = sequences[i]
        nd = ndim(seq)
        if nd == 3:
            pass
        elif nd < 3:
            sequences[i] = tf.expand_dims(sequences[i], nd)
        else:
            raise ValueError("Ndim too different to correct")

    def check(l):
        shapes = [shape(s) for s in l]
        # for now assume -1, can add axis argument later
        # check shapes match for concatenation
        compat = [ls for ls in shapes if ls[:-1] == shapes[0][:-1]]
        if len(compat) != len(shapes):
            raise ValueError("Tensors *must* be the same dim for now")

    check(sequences)
    check(outputs_info)

    seqs_shapes = [shape(s) for s in sequences]
    nd = len(seqs_shapes[0])
    seq_pack = tf.concat(nd - 1, sequences)
    outs_shapes = [shape(o) for o in outputs_info]
    nd = len(outs_shapes[0])
    init_pack = tf.concat(nd - 1, outputs_info)

    assert len(shape(seq_pack)) == 3
    assert len(shape(init_pack)) == 2

    def s_e(shps):
        starts = []
        ends = []
        prev_shp = 0
        for n, shp in enumerate(shps):
            start = prev_shp
            end = start + shp[-1]
            starts.append(start)
            ends.append(end)
            prev_shp = end
        return starts, ends

    # TF puts the initializer in front?
    def fnwrap(initializer, elems):
        starts, ends = s_e(seqs_shapes)
        sliced_elems = [elems[:, start:end] for start, end in zip(starts, ends)]
        starts, ends = s_e(outs_shapes)
        sliced_inits = [initializer[:, start:end]
                        for start, end in zip(starts, ends)]
        t = []
        t.extend(sliced_elems)
        t.extend(sliced_inits)
        # elems first then inits
        outs = fn(*t)
        nd = len(outs_shapes[0])
        outs_pack = tf.concat(nd - 1, outs)
        return outs_pack

    r = tf.scan(fnwrap, seq_pack, initializer=init_pack)

    if len(outs_shapes) > 1:
        starts, ends = s_e(outs_shapes)
        o = [r[:, :, start:end] for start, end in zip(starts, ends)]
        return o
    else:
        return r


def broadcast(original, size_to_match):
    nd = ndim(size_to_match)
    if nd == 3:
        b = tf.ones_like(size_to_match)[:, :, 0]
        b = tf.expand_dims(b, 2)
    else:
        raise ValueError("Unsupported ndim")
    # Assume axis 0 for now...
    bo = tf.expand_dims(original, 0)
    return b + bo


def shift(a, fill_value=0.):
    # shift forward, prepending fill_value
    nd = ndim(a)
    if nd == 3:
        pre = tf.zeros_like(a)[0:1, :, :] + fill_value
    elif nd == 2:
        pre = tf.zeros_like(a)[0:1, :] + fill_value
    else:
        raise ValueError("Unhandled ndim")
    ao = tf.concat(0, [pre, a])
    return ni_slice(ao, -1)
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
        (-1, 1) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = .9 * X
    X = X * 2 ** 15
    return X.astype('int16')


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


class base_iterator(object):
    # base class don't use directly
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
    # For "list of arrays" data
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

# Taken from
# http://stackoverflow.com/a/27518377
# A cool generator for counting the total number of lines in a file


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

if sys.version_info >= (3, 0):
    def raw_gen_count(filename):
        f = open(filename, 'rb')
        f_gen = _make_gen(f.raw.read)
        return sum(buf.count(b'\n') for buf in f_gen)
else:
    def raw_gen_count(filename):
        f = open(filename, 'rb')
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)


class character_file_iterator(object):
    def __init__(self, file_path, minibatch_size, start_index=0,
                 stop_index=np.inf, make_mask=True,
                 new_line_new_sequence=False,
                 sequence_length=None,
                 randomize=True, preprocess=None,
                 preprocess_kwargs={}):
        """
        Supports regular int, negative indexing, or float for setting
        stop_index
        Two "modes":
            new_line_new_sequence will do variable length minibatches based
            on newlines in the file

            without new_line_new_sequence the file is effectively one continuous
            stream, and minibatches will be sequence_length, batch_size, 1
        """
        self.minibatch_size = minibatch_size
        self.new_line_new_sequence = new_line_new_sequence
        self.sequence_length = sequence_length
        # this is weird and bad - holding file open?
        self.file_handle = open(file_path, mode="r")
        if randomize:
            self.random_state = np.random.RandomState(2177)
        self.make_mask = make_mask
        ext = file_path.split(".")[-1]
        if new_line_new_sequence and sequence_length is None:
            raise ValueError("sequence_length must be provided if",
                             "new_line_new_sequence is False!")
        # handle extensions like .gz?
        if ext == "txt":
            def _read():
                if self.new_line_new_sequence:
                    d = self.file_handle.readline()
                else:
                    d = self.file_handle.read(sequence_length)
                return d
        else:
            raise ValueError("Unhandled extension %s" % ext)
        self._read_file = _read
        vocabulary = "abcdefghijklmnopqrstuvwxyz"
        vocabulary += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        vocabulary += "1234567890"
        vocabulary += ".?!,;:'()-&*#$%@+_="
        vocabulary += "\n\t "
        self.vocabulary_size = len(vocabulary)
        v2k = {k: v for v, k in enumerate(vocabulary)}
        k2v = {v: k for k, v in v2k.items()}

        def tf(x):
            lu = [[v2k[xii] for xii in xi] for xi in x]
            # assumes even...
            r = np.array([np.array(lui, dtype="float32")[:, None] for lui in lu])
            return r.transpose(1, 0, 2)

        self.transform = tf

        def itf(x):
            x = x.transpose(1, 0, 2)
            return ["".join([k2v[int(xii.ravel())] for xii in xi]) for xi in x]

        self.inverse_transform = itf

        if self.new_line_new_sequence:
            _len = raw_gen_count(file_path)
        else:
            # THIS ASSUMES 1 BYTE PER CHAR
            # LETS PRETEND UTF8 IS NOT A THING
            _len = os.path.getsize(file_path)
        if stop_index >= 1:
            self.stop_index = int(min(stop_index, _len))
        elif stop_index > 0:
            # percentage
            self.stop_index = int(stop_index * _len)
        elif stop_index < 0:
            # negative index - must be int!
            self.stop_index = _len + int(stop_index)

        self.start_index = start_index
        if start_index != 0:
            raise ValueError("Currently unable to handle start point != 0")

        """
        if start_index < 0:
            # negative indexing
            self.start_index = _len + start_index
        elif start_index < 1:
            # float
            self.start_index = int(start_index * _len)
        else:
            # regular
            self.start_index = int(start_index)
        """
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
        if self.new_line_new_sequence:
            raise ValueError("BROKEN, NEEDS FIX")
            e = self._current_index + self.minibatch_size
            if e > self.stop_index:
                raise StopIteration("End of character file iterator reached!")
            data = [self._read_file() for fp in range(s, e)]
            data = [np.array(self.preprocess_function(d), dtype="float32")[:, None]
                    for d in data]

            li = list_iterator([data], self.minibatch_size, axis=1,
                               start_index=0,
                               stop_index=len(data), make_mask=self.make_mask)
            res = next(li)
            self._current_index = e
            return res
        else:
            e = s + self.minibatch_size * self.sequence_length
            if e > self.stop_index:
                raise StopIteration("End of character file iterator reached!")
            data = [self._read_file() for fp in range(self.minibatch_size)]
            data = self.transform(data)
            li = list_iterator([data.transpose(1, 0, 2)], self.minibatch_size, axis=1, start_index=0,
                               stop_index=len(data), make_mask=self.make_mask)
            res = next(li)
            self._current_index = e
            return res

    def reset(self):
        self._current_index = self.start_index
        self.file_handle.seek(0)


def file_to_piano_roll(file_path, sample_dur=None):
    from music21 import converter, graph, pitch, meter, tempo, stream
    pf = converter.parse(file_path)
    g = graph.PlotHorizontalBarPitchSpaceOffset(pf)
    durs = [float(repr(pf.flat.notes[i].duration).split(" ")[-1][:-1])
            for i in range(len(pf.flat.notes))]
    if sample_dur is not None:
        min_dur = min(durs)
    else:
        min_dur = sample_dur

    # hacked together from music21 mailing list
    # https://groups.google.com/forum/#!topic/music21list/1mMMrzHUUTk
    def graph_to_pianoroll(graph_obj, window_size=min_dur / 4):
        data_matrix = []
        for item in graph_obj.data:
            if item[0] != "":
                for subitem in item[1]:
                    data_matrix.append((item[0], subitem[0], subitem[1]))
        data_matrix = np.array(data_matrix)
        last_bar = max(data_matrix[:, 1])

        sr = 1. / window_size
        time_size = int((float(last_bar) + 1) * sr)
        if time_size < 1:
            raise ValueError("Window size too small!")
        # 128? Why...
        piano_roll = np.zeros((128, time_size))
        for event in data_matrix:
            p = event[0]
            p = p.replace("\\", "")
            p = p.replace("$", "")
            start_idx = int(float(event[1]) * sr)
            stop_idx = start_idx + int(float(event[2]) * sr)
            piano_roll[pitch.Pitch(p).midi, start_idx:stop_idx] = 1
        return piano_roll
    pr = graph_to_pianoroll(g)
    return pr, min_dur


# modified from
# https://raw.githubusercontent.com/hexahedria/biaxial-rnn-music-composition/master/midi_to_statematrix.py
# Original license: BSD 2-Clause
def piano_roll_to_midi(filename, piano_roll):
    import midi

    state_matrix = np.asarray(piano_roll)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    tickscale = 16

    #TODO: add decay? or look into a smarter synthesizer
    last_cmd_ti = 0
    prev_state = np.zeros_like(piano_roll[0])
    for ti, state in enumerate(state_matrix):
        off_notes = []
        on_notes = []
        deltas = state - prev_state
        deltas = deltas[1:]
        up = np.where(deltas > 0)[0]
        down = np.where(deltas < 0)[0]

        for i in down:
            off_notes.append(i)

        for i in up:
            on_notes.append(i)

        for note in off_notes:
            off_event_tick = (ti - last_cmd_ti) * tickscale
            off_event_pitch = note + 0
            track.append(midi.NoteOffEvent(tick=off_event_tick,
                                           pitch=off_event_pitch))
            last_cmd_ti = ti
        for note in on_notes:
            on_event_tick = (ti - last_cmd_ti) * tickscale
            v = 40
            on_event_pitch = note + 0
            track.append(midi.NoteOnEvent(tick=on_event_tick,
                                          velocity=v,
                                          pitch=on_event_pitch))
            last_cmd_ti = ti
        prev_state = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile(filename, pattern)


def notes_to_midi(filename, notes):
    # + 1 to simplify silence
    piano_roll = np.zeros((len(notes), 88 + 1)).astype("float32")
    for n in range(len(notes)):
        idx = notes[n].astype("int32")
        piano_roll[n, idx] = 1.
    # cut off silence
    piano_roll = piano_roll[:, 1:]
    piano_roll_to_midi(filename, piano_roll)


class midi_file_iterator(object):
    def __init__(self, files_path, minibatch_size, start_index=0,
                 stop_index=np.inf, make_mask=True,
                 new_file_new_sequence=False,
                 sequence_length=None,
                 randomize=True, preprocess=None,
                 preprocess_kwargs={}):
        """
        Supports regular int, negative indexing, or float for setting
        stop_index
        Two "modes":
            new_line_new_sequence will do variable length minibatches based
            on newlines in the file

            without new_line_new_sequence the file is effectively one continuous
            stream, and minibatches will be sequence_length, batch_size, 1
        """
        # how to get time signature? files corrupted
        # ts = pf.getElementsByClass(meter.TimeSignature)[0]
        # met = pf.getElementsByClass(tempo.MetronomeMark)[0]

        self.minibatch_size = minibatch_size
        self.new_file_new_sequence = new_file_new_sequence
        self.sequence_length = sequence_length
        if randomize:
            self.random_state = np.random.RandomState(2177)
        self.make_mask = make_mask
        if new_file_new_sequence and sequence_length is None:
            raise ValueError("sequence_length must be provided if",
                             "new_line_new_sequence is False!")

        files = glob.glob(files_path)

        # Ew, hardcode bach for now...
        a = [file_to_piano_roll(f, sample_dur=1.25) for f in files]
        min_durs = [aa[1] for aa in a]
        piano_rolls = [aa[0] for aa in a]

        if new_file_new_sequence:
            raise ValueError("Unhandled case")
        else:
            data = np.concatenate(piano_rolls, axis=-1)

        if new_file_new_sequence:
            raise ValueError("Unhandled case")
        else:
            _len = data.shape[-1]
        if stop_index >= 1:
            self.stop_index = int(min(stop_index, _len))
        elif stop_index > 0:
            # percentage
            self.stop_index = int(stop_index * _len)
        elif stop_index < 0:
            # negative index - must be int!
            self.stop_index = _len + int(stop_index)

        self._data = data
        self.vocabulary_size = 128 + 1  # + 1 for silence
        # set automatically
        # self.simultaneous_notes = int(max(np.sum(self._data, axis=0)))
        self.simultaneous_notes = 4

        self.start_index = start_index
        if start_index != 0:
            raise ValueError("Currently unable to handle start point != 0")

        """
        if start_index < 0:
            # negative indexing
            self.start_index = _len + start_index
        elif start_index < 1:
            # float
            self.start_index = int(start_index * _len)
        else:
            # regular
            self.start_index = int(start_index)
        """
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
        if self.new_file_new_sequence:
            raise ValueError("not handled")
        else:
            e = s + self.minibatch_size * self.sequence_length
            if e > self.stop_index:
                raise StopIteration("End of file iterator reached!")
            data = self._data[:, s:e]
            shp = data.shape
            data = data.reshape((shp[0], self.minibatch_size, -1)).transpose(2, 1, 0)
            li = list_iterator([data.transpose(1, 0, 2)], self.minibatch_size,
                                axis=1, start_index=0,
                                stop_index=len(data), make_mask=self.make_mask)
            res = next(li)
            # embedding elements directly?
            # might make sense to embed based on relative context...
            # get top self.simultaneous_notes items from piano roll
            shp = res[0].shape
            keys = np.zeros((shp[0], shp[1], self.simultaneous_notes),
                             dtype="float32")
            for i in range(shp[0]):
                r = res[0][i]
                elem, idx = np.where(r > 0)
                for el in np.unique(elem):
                    fi = np.where(elem == el)[0]
                    k = idx[fi] + 1  # + 1 so silence is always 0
                    # In case we decide to cut off some notes
                    k = k[:self.simultaneous_notes]
                    """
                    # Now that we have some notes, do "interval encoding"
                    # notewise, skipping sil
                    mi = min(k[k > 0])
                    k[k > mi] = k[k > mi] - mi
                    # Make sure all the key values are > 0
                    assert np.all(k >= 0)
                    """
                    keys[i, el, :len(k)] = k
            self._current_index = e
            return keys, res[1]

    def reset(self):
        self._current_index = self.start_index


def duration_and_pitch_to_midi(filename, durations, pitches, prime_until=0):
    """
    durations and pitches should both be 2D
    [time_steps, n_notes]
    """

    from magenta.protobuf import music_pb2
    sequence = music_pb2.NoteSequence()

    """
    from magenta.lib.note_sequence_io import note_sequence_record_iterator
    reader = note_sequence_record_iterator('BachChorales.tfrecord')
    ns = reader.next()
    ti = tfrecord_iterator("BachChorales.tfrecord", 50, make_mask=True,
                           sequence_length=50)
    """

    # Hardcode for now, eventually randomize?
    # or predict...
    sequence.ticks_per_beat = 480
    ts = sequence.time_signatures.add()
    ts.time = 1.0
    ts.numerator = 4
    ts.denominator = 4

    ks = sequence.key_signatures.add()
    ks.key = 0
    ks.mode = ks.MAJOR

    tempos = sequence.tempos.add()
    tempos.bpm = 120
    # ti.simultaneous_notes
    sn = 4
    # ti.time_classes
    # this should be moved out for sure - this is really duration class, pitch!
    time_classes = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

    dt = copy.deepcopy(durations)
    for n in range(len(time_classes)):
        dt[dt == n] = time_classes[n]

    # why / 8?
    delta_times = [dt[..., i] / 8 for i in range(sn)]
    end_times = [delta_times[i].cumsum(axis=0) for i in range(sn)]
    start_times = [end_times[i] - delta_times[i] for i in range(sn)]
    voices = [pitches[..., i] for i in range(sn)]

    midi_notes = []
    default_instrument = 0
    default_program = 0
    priming_instrument = 79
    priming_program = 79
    sequence.total_time = float(max([end_times[i][-1] for i in range(sn)]))

    assert len(delta_times[0]) == len(voices[0])
    for n in range(len(delta_times[0])):
        for i in range(len(voices)):
            # Hardcode 1 sample for now
            v = voices[i][n]
            s = start_times[i][n]
            e = end_times[i][n]
            if v != 0.:
                # Skip silence voices... for now
                # namedtuple?
                if n >= prime_until:
                    midi_notes.append((default_instrument, default_program, v, s, e))
                else:
                    midi_notes.append((priming_instrument, priming_program, v, s, e))
    for tup in midi_notes:
        sequence_note = sequence.notes.add()
        i = tup[0]
        p = tup[1]
        v = tup[2]
        s = tup[3]
        e = tup[4]
        sequence_note.instrument = int(i)
        sequence_note.program = int(p)
        sequence_note.pitch = int(v)
        sequence_note.velocity = int(127.)
        sequence_note.start_time = float(s)
        sequence_note.end_time = float(e)

    from magenta.lib.midi_io import sequence_proto_to_pretty_midi
    pretty_midi_object = sequence_proto_to_pretty_midi(sequence)
    pretty_midi_object.write(filename)


class tfrecord_duration_and_pitch_iterator(object):
    def __init__(self, files_path, minibatch_size, start_index=0,
                 stop_index=np.inf, make_mask=False,
                 make_augmentations=False,
                 new_file_new_sequence=False,
                 sequence_length=None,
                 randomize=True, preprocess=None,
                 preprocess_kwargs={}):
        """
        Supports regular int, negative indexing, or float for setting
        stop_index
        Two "modes":
            new_line_new_sequence will do variable length minibatches based
            on newlines in the file

            without new_line_new_sequence the file is effectively one continuous
            stream, and minibatches will be sequence_length, batch_size, 1
        """
        filename_queue = tf.train.string_input_producer([files_path])
        # TODO: FIX THIS HANDCRAFTED WEB OF LIES
        # Need to figure out magenta path stuff later
        # for now...
        # PYTHONPATH=$PYTHONPATH:$HOME/src/magenta/
        # bazel build //magenta:protobuf:music_py_pb2
        # symlink bazel-out/local-opt/genfiles/magenta/protobuf/music_pb2.py
        # into protobuf dir.
        # Add __init__ files all over the place
        # symlinked the BachChorale data in for now too...
        from magenta.lib.note_sequence_io import note_sequence_record_iterator
        reader = note_sequence_record_iterator(files_path)
        all_ds = []
        all_ps = []
        self.note_classes = list(np.arange(88 + 1))  # + 1 for silence
        # set automatically
        # self.simultaneous_notes = int(max(np.sum(self._data, axis=0)))
        self.simultaneous_notes = 4
        for ns in reader:
            notes = ns.notes
            st = np.array([n.start_time for n in notes]).astype("float32")
            et = np.array([n.end_time for n in notes]).astype("float32")
            dt = et - st
            p = np.array([n.pitch for n in notes]).astype("float32")

            sample_times = sorted(list(set(st)))
            # go straight for pitch and delta time encoding
            sn = self.simultaneous_notes
            pitch_slices = [p[st == sti][::-1] for sti in sample_times]
            # This monster fills in 0s so that array size is consistent
            pitch_slices = [p[:sn] if len(p) >= sn
                            else
                            np.concatenate((p, np.array([0.] * (sn - len(p)),
                                                        dtype="float32")))
                            for p in pitch_slices]
            start_slices = [st[st == sti] for sti in sample_times]
            end_slices = [et[st == sti] for sti in sample_times]
            start_slices = [ss[:sn] if len(ss) >= sn
                            else
                            np.concatenate((ss, np.array([ss[0]] * (sn - len(ss)),
                                                        dtype="float32")))
                            for ss in start_slices]
            end_slices = [es[:sn] if len(es) >= sn
                          else
                          np.concatenate((es, np.array([max(es)] * (sn - len(es)),
                                                        dtype="float32")))
                          for es in end_slices]
            start_slices = np.array(start_slices)
            end_slices = np.array(end_slices)
            delta_slices = end_slices - start_slices
            maxlen = max([len(ps) for ps in pitch_slices])
            all_ds.append(np.array(delta_slices))
            all_ps.append(np.array(pitch_slices))
        max_seq = max([len(ds) for ds in all_ds])
        min_seq = min([len(ds) for ds in all_ds])
        assert len(all_ds) == len(all_ps)
        if new_file_new_sequence:
            raise ValueError("Unhandled case")
        else:
            """
            self.time_classes = list(np.unique(np.concatenate(all_ds).ravel()))
            ds0 = all_ds[0]
            for n, i in enumerate(self.time_classes):
                ds0[ds0 == i] = n
            duration_and_pitch_to_midi("truf_b.mid", all_ds[0], all_ps[0])
            """

            if not make_augmentations:
                all_ds = np.concatenate(all_ds)
                all_ps = np.concatenate(all_ps)
            else:
                new_ps_list = []
                new_ds_list = []
                assert len(all_ds) == len(all_ps)
                for n, (ds, ps) in enumerate(zip(all_ds, all_ps)):
                    new_ps_list.append(ps)
                    new_ds_list.append(ds)
                    # Do +- 5 steps for all 11 offsets
                    for i in range(5):
                        new_up = ps + i
                        # Put silences back
                        new_up[new_up == i] = 0.
                        # Edge case... shouldn't come up in general
                        new_up[new_up > 88] = 88.
                        new_down = ps - i
                        # Put silences back
                        new_down[new_down == -i] = 0.
                        # Edge case... shouldn't come up in general
                        new_down[new_down < 0.] = 1.

                        new_ps_list.append(new_up)
                        new_ds_list.append(ds)

                        new_ps_list.append(new_down)
                        new_ds_list.append(ds)
                all_ds = np.concatenate(new_ds_list)
                all_ps = np.concatenate(new_ps_list)

            """
            self.time_classes = list(np.unique(np.concatenate(all_ds).ravel()))
            ds0 = all_ds[:400]
            for n, i in enumerate(self.time_classes):
                ds0[ds0 == i] = n

            duration_and_pitch_to_midi("truf_nrs.mid", ds0, all_ps[:400])
            """
            self._min_time_data = np.min(all_ds)
            self._max_time_data = np.max(all_ds)
            self.time_classes = list(np.unique(all_ds.ravel()))

            truncate = len(all_ds) - len(all_ds) % minibatch_size
            all_ds = all_ds[:truncate]
            all_ps = all_ps[:truncate]

            """
            all_ds = all_ds.reshape(len(all_ds) // minibatch_size,
                                    minibatch_size, -1)
            all_ps = all_ps.reshape(len(all_ps) // minibatch_size,
                                    minibatch_size, -1)
            """

            # transpose necessary to preserve data structure!
            all_ds = all_ds.transpose(1, 0)
            all_ds = all_ds.reshape(-1, minibatch_size,
                                    all_ds.shape[1] // minibatch_size)
            all_ds = all_ds.transpose(2, 1, 0)
            all_ps = all_ps.transpose(1, 0)
            all_ps = all_ps.reshape(-1, minibatch_size,
                                    all_ps.shape[1] // minibatch_size)
            all_ps = all_ps.transpose(2, 1, 0)

            """
            self.time_classes = list(np.unique(np.concatenate(all_ds).ravel()))
            ds0 = all_ds[:, 0]
            for n, i in enumerate(self.time_classes):
                ds0[ds0 == i] = n

            duration_and_pitch_to_midi("truf_rs.mid", ds0, all_ps[:, 0])
            """

            _len = len(all_ds)
            self._time_data = all_ds
            self._pitch_data = all_ps

        self.minibatch_size = minibatch_size
        self.new_file_new_sequence = new_file_new_sequence
        self.sequence_length = sequence_length
        if randomize:
            self.random_state = np.random.RandomState(2177)
        self.make_mask = make_mask
        if new_file_new_sequence and sequence_length is None:
            raise ValueError("sequence_length must be provided if",
                             "new_line_new_sequence is False!")

        if stop_index >= 1:
            self.stop_index = int(min(stop_index, _len))
        elif stop_index > 0:
            # percentage
            self.stop_index = int(stop_index * _len)
        elif stop_index < 0:
            # negative index - must be int!
            self.stop_index = _len + int(stop_index)


        self.start_index = start_index
        if start_index < 0:
            # negative indexing
            self.start_index = _len + start_index
        elif start_index < 1:
            # float
            self.start_index = int(start_index * _len)
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
        if self.new_file_new_sequence:
            raise ValueError("not handled")
        else:
            e = s + self.sequence_length
            if e > self.stop_index:
                raise StopIteration("End of file iterator reached!")
            time_data = self._time_data[s:e]
            for n, i in enumerate(self.time_classes):
                # turn them into duration classes
                time_data[time_data == i] = n
            time_data = time_data
            pitch_data = self._pitch_data[s:e]

            if self.make_mask is False:
                res = (time_data, pitch_data)
            else:
                raise ValueError("Unhandled mask making")
                # super lazy way to make a mask
                li = list_iterator([time_data.transpose(1, 0, 2),
                                    pitch_data.transpose(1, 0, 2)], self.minibatch_size,
                                    axis=1, start_index=0,
                                    stop_index=len(time_data), make_mask=self.make_mask)
                res = next(li)
                # embedding elements directly?
                # might make sense to embed based on relative context...
                # get top self.simultaneous_notes items from piano roll
                shp = res[0].shape
            self._current_index = e
            return res

    def reset(self):
        self._current_index = self.start_index


def notesequence_to_piano_roll(notesequence, sample_rate=0.0625):
    ns = notesequence
    notes = ns.notes
    st = np.array([n.start_time for n in notes]).astype("float32")
    et = np.array([n.end_time for n in notes]).astype("float32")
    dt = et - st
    sample_steps = dt / sample_rate
    # If this fails the sample_rate is not an integer divisor
    assert np.all(np.abs(sample_steps - sample_steps.astype("int32")) < 1E-8)

    sample_times = sorted(list(set(st)))
    # TODO: generalize this
    sn = 4

    p = np.array([n.pitch for n in notes]).astype("float32")
    pitch_slices = [p[st == sti][::-1][:sn] for sti in sample_times]
    start_slices = [st[st == sti][::-1][:sn] for sti in sample_times]
    end_slices = [et[st == sti][::-1][:sn] for sti in sample_times]
    # If this fails something bad has happened in the ragged reshape
    assert len(pitch_slices) == len(start_slices)
    assert len(start_slices) == len(end_slices)
    group = zip(pitch_slices, start_slices, end_slices)

    # 88 + 1 for every note + each voice silence....
    maxlen = int(max(et) / sample_rate) + 1
    piano_roll = np.zeros((maxlen, 88 + 1)).astype("float32")

    # This iteration is terrible but making it faster will be annoying
    current_time = 0.
    for ti in range(len(piano_roll)):
        for ps, ss, es in group:
            if np.all(es < current_time):
                continue
            smask = ss <= current_time
            emask = es >= current_time
            mask = smask * emask
            idx = ps.astype("int32") * mask
            piano_roll[ti, idx] = 1.
            if np.all(es > current_time):
                # no need to iterate all
                break
        current_time += sample_rate
    # Cut off silence, handle turning to indices or padding in iterator!
    piano_roll = piano_roll[:, 1:]
    return piano_roll


# TODO: Eliminate duplicate stuff...
class tfrecord_roll_iterator(object):
    def __init__(self, files_path, minibatch_size, start_index=0,
                 stop_index=np.inf, make_mask=False,
                 new_file_new_sequence=False,
                 sequence_length=None,
                 randomize=True, preprocess=None,
                 preprocess_kwargs={}):
        """
        Supports regular int, negative indexing, or float for setting
        stop_index
        Two "modes":
            new_line_new_sequence will do variable length minibatches based
            on newlines in the file

            without new_line_new_sequence the file is effectively one continuous
            stream, and minibatches will be sequence_length, batch_size, 1
        """
        # TODO: FIX THIS HANDCRAFTED WEB OF LIES
        # Need to figure out magenta path stuff later
        # for now...
        # PYTHONPATH=$PYTHONPATH:$HOME/src/magenta/
        # bazel build //magenta:protobuf:music_py_pb2
        # symlink bazel-out/local-opt/genfiles/magenta/protobuf/music_pb2.py
        # into protobuf dir.
        # Add __init__ files all over the place
        # symlinked the BachChorale data in for now too...
        from magenta.lib.note_sequence_io import note_sequence_record_iterator
        reader = note_sequence_record_iterator(files_path)
        self.note_classes = list(np.arange(88 + 1))  # + 1 for silence
        # set automatically
        # self.simultaneous_notes = int(max(np.sum(self._data, axis=0)))
        self.simultaneous_notes = 4
        #import multiprocessing as mp
        #pool = mp.Pool()
        #all_pr = pool.map(notesequence_to_piano_roll, reader)
        #del pool
        all_pr = map(notesequence_to_piano_roll, reader)

        if new_file_new_sequence:
            raise ValueError("Unhandled case")
        else:
            all_pr = np.concatenate(all_pr, axis=0)
            truncate = len(all_pr) - len(all_pr) % minibatch_size
            all_pr = all_pr[:truncate]
            # transpose necessary to preserve data structure!
            all_pr = all_pr.transpose(1, 0)
            all_pr = all_pr.reshape(-1, minibatch_size,
                                    all_pr.shape[1] // minibatch_size)
            all_pr = all_pr.transpose(2, 1, 0)
            _len = len(all_pr)
            self._data = all_pr

        self.minibatch_size = minibatch_size
        self.new_file_new_sequence = new_file_new_sequence
        self.sequence_length = sequence_length
        if randomize:
            self.random_state = np.random.RandomState(2177)
        self.make_mask = make_mask
        if new_file_new_sequence and sequence_length is None:
            raise ValueError("sequence_length must be provided if",
                             "new_line_new_sequence is False!")

        if stop_index >= 1:
            self.stop_index = int(min(stop_index, _len))
        elif stop_index > 0:
            # percentage
            self.stop_index = int(stop_index * _len)
        elif stop_index < 0:
            # negative index - must be int!
            self.stop_index = _len + int(stop_index)


        self.start_index = start_index
        if start_index < 0:
            # negative indexing
            self.start_index = _len + start_index
        elif start_index < 1:
            # float
            self.start_index = int(start_index * _len)
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
        if self.new_file_new_sequence:
            raise ValueError("not handled")
        else:
            e = s + self.sequence_length
            if e > self.stop_index:
                raise StopIteration("End of file iterator reached!")

            # Convert from raw piano roll to embedding elements
            data = self._data[s:e]
            dr = data.reshape(-1, data.shape[-1])
            note_data = np.zeros_like(dr[:, :self.simultaneous_notes])
            for i in range(len(dr)):
                # Truncate to at most N voices
                notes = dr[i].nonzero()[0][:self.simultaneous_notes]
                note_data[i, :len(notes)] = notes
            note_data = note_data.reshape((data.shape[0], data.shape[1], -1))
            if self.make_mask is False:
                res = note_data
            else:
                raise ValueError("Not handled")
                # super lazy way to make a mask
                li = list_iterator([note_data,],
                                   self.minibatch_size,
                                   axis=1, start_index=0,
                                   stop_index=len(note_data),
                                   make_mask=self.make_mask)
                res = next(li)
            self._current_index = e
            return res

    def reset(self):
        self._current_index = self.start_index


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
    return np.zeros(shape).astype("float32")


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
    return np.ones(shape).astype("float32")


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
        "float32")


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
    return (scale * random_state.randn(*shp)).astype("float32")


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
                                size=shp).astype("float32")


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
    return var * random_state.randn(*shp).astype("float32")


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
        "float32")


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
    return std * random_state.randn(*shp).astype("float32")


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
        "float32")


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
    return std * random_state.randn(*shp).astype("float32")


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
    return (scale * res).astype("float32")


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
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims):
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default"):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element retuTrue
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
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


def LearnedInitHidden(list_of_inputs, list_of_shapes):
    raise ValueError("Need rewriting to TF")
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


def OneHot(indices, n_symbols):
    shp = shape(indices)
    if shp[-1] == 1 and len(shp) == 3:
        indices = indices[:, :, 0]
    else:
        raise ValueError("Currently unnsuppported case in OneHot")
    oh = tf.one_hot(tf.cast(indices, "int32"), n_symbols,
                    dtype="float32", axis=-1)
    return oh


def Embedding(indices, n_symbols, output_dim, random_state, name=None):
    """
    Last dimension of indices tensor must be 1!!!!
    """
    vectors = tf.Variable(random_state.randn(n_symbols, output_dim).astype("float32"),
                          trainable=True)
    ii = tf.cast(indices, "int32")
    shp = shape(ii)
    nd = len(shp)
    output_shape = [
        shp[i]
        for i in range(nd - 1)
    ] + [output_dim]
    lu = tf.nn.embedding_lookup(vectors, ii)
    if nd == 3:
        lu = lu[:, :, 0, :]
    else:
        return lu
    return lu


def Multiembedding(multi_indices, n_symbols, output_dim, random_state,
                   name=None):
    """
    Helper to compute many embeddings and concatenate

    Requires input indices to be 3D, with last axis being the "iteration" dimension
    """
    # Should n_symbols be a list of embedding values?
    output_embeds = []
    shp = shape(multi_indices)
    if len(shp) != 3:
        raise ValueError("Unhandled rank != 3 for input multi_indices")
    index_range = shp[-1]
    for i in range(index_range):
        e = Embedding(multi_indices[:, :, i], n_symbols, output_dim, random_state)
        output_embeds.append(e)
    return tf.concat(2, output_embeds)


def Automask(input_tensor, n_masks, axis=-1, name=None):
    """
    Auto masker to make multiple MADE/pixelRNN style masking easier

    n_masks *must* be an even divisor of input_tensor.shape[axis]

    masks are basically of form

    [:, :, :i * divisor_dim] = 1.
    for i in range(n_masks)

    a 1, 4 example with n_masks = 2 would be

    mask0 = [0., 0., 0., 0.]
    mask1 = [1., 1., 0., 0.]

    The goal of these masks is to model p(y_i,t | x_<=t, y_<i,<=t) in order
    to maximize the context a given prediction can "see".

    This function will return n_masks copies of input_tensor (* a mask) in list
    """
    if axis != -1:
        raise ValueError("Axis not currently supported!")
    shp = shape(input_tensor)
    shp_tup = tuple([1] * (len(shp) - 1) + [shp[-1]])
    div = shp[-1] // n_masks
    assert int(div * n_masks) == shp[-1]
    masks = [np.zeros(shp_tup).astype("float32") for i in range(n_masks)]
    if n_masks < 2:
        raise ValueError("unhandled small n_masks value")
    output_tensors = [input_tensor]
    for i in range(1, n_masks):
        masks[i][..., :i * div] = 1.
        output_tensors.append(masks[i] * input_tensor)
    return output_tensors


def Linear(list_of_inputs, input_dims, output_dim, random_state, name=None,
           init=None, scale="default", weight_norm=None, biases=True):
    """
    Can pass weights and biases directly if needed through init
    """
    if weight_norm is None:
        # Let other classes delegate to default of linear
        weight_norm = True
    # assume both have same shape -_-
    nd = ndim(list_of_inputs[0])
    input_var = tf.concat(concat_dim=nd - 1, values=list_of_inputs)
    input_dim = sum(input_dims)
    terms = []
    if (init is None) or (type(init) is str):
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        weight_values = init[0]
    weight = tf.Variable(weight_values, trainable=True)
    # http://arxiv.org/abs/1602.07868
    if weight_norm:
        norm_values = np.linalg.norm(weight_values, axis=0)
        norms = tf.Variable(norm_values, trainable=True)
        norm = tf.sqrt(tf.reduce_sum(tf.abs(weight ** 2), reduction_indices=[0],
                                     keep_dims=True))
        normed_weight = weight * (norms / norm)
        terms.append(dot(input_var, normed_weight))
    else:
        terms.append(dot(input_var, weight))

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim])
        else:
            b = init[1]
        biases = tf.Variable(b, trainable=True)
        terms.append(biases)
    out = reduce(lambda a, b: a + b, terms)
    # Not adding default names/scoping for now
    # out.name = get_generic_name() + ".output"
    return out


def make_conv_weights(in_dim, out_dims, kernel_size, random_state):
    raise ValueError("Need rewriting to TF")
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
    raise ValueError("Need rewriting to TF")
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
    raise ValueError("Need rewriting to TF")
    """
    Repeat unpooling
    """
    return input.repeat(pool_size[0], axis=2).repeat(pool_size[1], axis=3)


def conv2d_transpose(input, filters, biases=None, border_mode=0, stride=(1, 1)):
    raise ValueError("Need rewriting to TF")
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
    elif forward_init is None:
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
        mask=None, name=None, init=None, scale="default", weight_norm=None,
        biases=False):
        if init is None:
            hidden_init = "ortho"
        elif init == "normal":
            hidden_init = "normal"
        else:
            raise ValueError("Not yet configured for other inits")

        ndi = ndim(inp)
        if mask is None:
            if ndi == 2:
                mask = tf.ones_like(inp)
            else:
                raise ValueError("Unhandled ndim")

        ndm = ndim(mask)
        if ndm == (ndi - 1):
            mask = tf.expand_dims(mask, ndm - 1)

        _, _, Wur, U = gru_weights(input_dim, hidden_dim,
                                   hidden_init=hidden_init,
                                   random_state=random_state)
        dim = hidden_dim
        f1 = Linear([previous_state], [2 * hidden_dim], 2 * hidden_dim,
                    random_state, name=(name, "update/reset"), init=[Wur],
                    biases=biases)
        gates = sigmoid(f1 + gate_inp)
        update = gates[:, :dim]
        reset = gates[:, dim:]
        state_reset = previous_state * reset
        f2 = Linear([state_reset], [hidden_dim], hidden_dim,
                    random_state, name=(name, "state"), init=[U], biases=biases)
        next_state = tf.tanh(f2 + inp)
        next_state = next_state * update + previous_state * (1. - update)
        next_state = mask * next_state + (1. - mask) * previous_state
        return next_state


def GRUFork(list_of_inputs, input_dims, output_dim, random_state, name=None,
            init=None, scale="default", weight_norm=None, biases=True):
        gates = Linear(list_of_inputs, input_dims, 3 * output_dim,
                       random_state=random_state,
                       name=(name, "gates"), init=init, scale=scale,
                       weight_norm=weight_norm,
                       biases=biases)
        dim = output_dim
        nd = ndim(gates)
        if nd == 2:
            d = gates[:, :dim]
            g = gates[:, dim:]
        elif nd == 3:
            d = gates[:, :, :dim]
            g = gates[:, :, dim:]
        else:
            raise ValueError("Unsupported ndim")
        return d, g


def lstm_weights(input_dim, hidden_dim, forward_init=None, hidden_init="normal",
                 random_state=None):
    if random_state is None:
        raise ValueError("Must pass random_state!")
    shape = (input_dim, hidden_dim)
    if forward_init == "normal":
        W = np.hstack([np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state)])
    elif forward_init == "fan":
        W = np.hstack([np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state)])
    elif forward_init is None:
        if input_dim == hidden_dim:
            W = np.hstack([np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state)])
        else:
            # lecun
            W = np.hstack([np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state)])
    else:
        raise ValueError("Unknown forward init type %s" % forward_init)
    b = np_zeros((4 * shape[1],))
    # Set forget gate bias to 1
    b[shape[1]:2 * shape[1]] += 1.

    if hidden_init == "normal":
        U = np.hstack([np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state),])
    elif hidden_init == "ortho":
        U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state), ])
    return W, b, U


def LSTM(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
         mask=None, name=None, init=None, scale="default", weight_norm=None,
         biases=False):
        """
        Output is the concatenation of hidden state and cell
        so 2 * hidden dim
        will need to slice yourself, or handle in some way
        This was done specifically to have the GRU, LSTM activations swappable
        """
        if gate_inp != "LSTMGates":
            raise ValueError("Use LSTMFork to setup this block")
        if init is None:
            hidden_init = "ortho"
        elif init == "normal":
            hidden_init = "normal"
        else:
            raise ValueError("Not yet configured for other inits")

        ndi = ndim(inp)
        if mask is None:
            if ndi == 2:
                mask = tf.ones_like(inp)[:, :hidden_dim]
            else:
                raise ValueError("Unhandled ndim")

        ndm = ndim(mask)
        if ndm == (ndi - 1):
            mask = tf.expand_dims(mask, ndm - 1)

        _, _, U = lstm_weights(input_dim, hidden_dim,
                               hidden_init=hidden_init,
                               random_state=random_state)
        dim = hidden_dim

        def _s(p, d):
           return p[:, d * dim:(d+1) * dim]

        previous_cell = _s(previous_state, 1)
        previous_st = _s(previous_state, 0)

        preactivation = Linear([previous_st], [4 * hidden_dim],
                                4 * hidden_dim,
                                random_state, name=(name, "preactivation"),
                                init=[U],
                                biases=False) + inp

        ig = sigmoid(_s(preactivation, 0))
        fg = sigmoid(_s(preactivation, 1))
        og = sigmoid(_s(preactivation, 2))
        cg = tanh(_s(preactivation, 3))

        cg = fg * previous_cell + ig * cg
        cg = mask * cg + (1. - mask) * previous_cell

        hg = og * tanh(cg)
        hg = mask * hg + (1. - mask) * previous_st

        next_state = tf.concat(1, [hg, cg])
        return next_state


def LSTMFork(list_of_inputs, input_dims, output_dim, random_state, name=None,
             init=None, scale="default", weight_norm=None, biases=True):
        """
        output dim should be the hidden size for each gate
        overall size will be 4x
        """
        inp_d = np.sum(input_dims)
        W, b, U = lstm_weights(inp_d, output_dim,
                               random_state=random_state)
        f_init = [W, b]
        inputs = Linear(list_of_inputs, input_dims, 4 * output_dim,
                        random_state=random_state,
                        name=(name, "inputs"), init=f_init, scale=scale,
                        weight_norm=weight_norm,
                        biases=True)
        return inputs, "LSTMGates"


def logsumexp(x, axis=None):
    raise ValueError("Need rewriting to TF")
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max),
                              axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def softmax(X):
    # should work for both 2D and 3D
    dim = len(shape(X))
    e_X = tf.exp(X - tf.reduce_max(X, reduction_indices=[dim - 1],
                                   keep_dims=True))
    out = e_X / tf.reduce_sum(e_X, reduction_indices=[dim - 1], keep_dims=True)
    return out


def numpy_softmax(X, temperature=1.):
    # should work for both 2D and 3D
    dim = X.ndim
    X = X / temperature
    e_X = np.exp((X - X.max(axis=dim - 1, keepdims=True)))
    out = e_X / e_X.sum(axis=dim - 1, keepdims=True)
    return out


def elu(x, alpha=1):
    raise ValueError("Need rewriting to TF")
    """
    Compute the element-wise exponential linear activation function.
    From theano 0.0.8 - here for backwards compat
    """
    return tensor.switch(x > 0, x, alpha * (tensor.exp(x) - 1))


def relu(x):
    raise ValueError("Need rewriting to TF")
    return x * (x > 1e-6)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.sigmoid(x)


def binary_crossentropy(predicted_values, true_values):
    raise ValueError("Need rewriting to TF")
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
    raise ValueError("TFIFY")
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def categorical_crossentropy(predicted_values, true_values, class_weights=None,
                             eps=None):
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

    eps : float, default None
        Epsilon to be added during log calculation to avoid NaN values.

    class_weights : dictionary with form {class_index: weight)
        Unspecified classes will get the default weight of 1.
        See discussion here for understanding how class weights work
        http://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    if eps != None:
        raise ValueError("Not yet implemented")
    else:
        predicted_values = tf.to_float(predicted_values)
        true_values = tf.to_float(true_values)
    tshp = shape(true_values)
    pshp = shape(predicted_values)
    if tshp[-1] == 1 or len(tshp) < len(pshp):
        logger.info("True values dimension should match predicted!")
        logger.info("Expected %s, got %s" % (pshp, tshp))
        if tshp[-1] == 1:
            # squeeze out the last dimension
            logger.info("Removing last dimension of 1 from %s" % str(tshp))
            if len(tshp) == 3:
                true_values = true_values[:, :, 0]
            elif len(tshp) == 2:
                true_values = true_values[:, 0]
            else:
                raise ValueError("Unhandled dimensions in squeeze")
        tshp = shape(true_values)
        if len(tshp) == (len(pshp) - 1):
            logger.info("Changing %s to %s with one hot encoding" % (tshp, pshp))
            tf.cast(true_values, "int32")
            ot = tf.one_hot(tf.cast(true_values, "int32"), pshp[-1],
                            dtype="float32", axis=-1)
            true_values = ot
        elif len(tshp) == len(pshp):
            pass
        else:
            raise ValueError("Dimensions of true_values and predicted_values"
                             "mismatched")
        # repeat so the right shape is captured
        tshp = shape(true_values)
    cw = np.ones(pshp[-1], dtype="float32")
    if class_weights is not None:
        for k, v in class_weights.items():
            cw[k] = v
        cw = cw / np.sum(cw)
        # np.sum() cw really should be close to 1
        cw = cw / (np.sum(cw) + 1E-12)
    # expand dimensions for broadcasting
    if len(tshp) == 3:
        cw = cw[None, None, :]
    elif len(tshp) == 2:
        cw = cw[None, :]
    nd = len(shape(true_values))
    assert nd == len(shape(predicted_values))
    stable_result = tf.select(true_values < 1E-20, 0. * predicted_values,
                              cw * true_values * tf.log(predicted_values))
    ce = -tf.reduce_sum(stable_result, reduction_indices=[nd - 1])
    return ce


def sample_binomial(coeff, n_bins, theano_rng, debug=False):
    raise ValueError("Need rewriting to TF")
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
    return tensor.cast(idx, "float32")


def sample_softmax(coeff, theano_rng, epsilon=1E-5, debug=False):
    raise ValueError("Need rewriting to TF")
    if coeff.ndim > 2:
        raise ValueError("Unsupported dim")
    if debug:
        idx = coeff.argmax(axis=1)
    else:
        idx = tensor.argmax(theano_rng.multinomial(pvals=coeff, dtype=coeff.dtype),
                            axis=1)
    return tensor.cast(idx, "float32")


def numpy_sample_softmax(coeff, random_state, class_weights=None, debug=False):
    """
    Numpy function to sample from a softmax distribution

    Parameters
    ----------
    coeff : array-like, shape 2D or higher
        The predicted class probabilities out of some layer,
        normally the output of a softmax

    random_state : numpy.random.RandomState() instance

    class_weights : dictionary with form {class_index: weight}, default None
        Unspecified classes will get the default weight of 1.
        See discussion here for understanding how class weights work
        http://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work

    debug: Boolean, default False
        Take the argmax instead of sampling. Useful for debugging purposes or
        testing greedy sampling.

    Returns
    -------
    samples : array-like, shape of coeff.shape[:-1]
        Sampled values
    """
    reshape_dims = coeff.shape[:-1]
    coeff = coeff.reshape((-1, coeff.shape[-1]))
    cw = np.ones((1, coeff.shape[-1])).astype("float32")
    if class_weights is not None:
        for k, v in class_weights.items():
            cw[k] = v
        cw = cw / np.sum(cw)
        cw = cw / (np.sum(cw) + 1E-12)
    if debug:
        idx = coeff.argmax(axis=-1)
    else:
        coeff = cw * coeff
        # renormalize to avoid numpy errors about summation...
        # end result shouldn't change
        coeff = coeff / (coeff.sum(axis=1, keepdims=True) + 1E-3)
        idxs = [np.argmax(random_state.multinomial(1, pvals=coeff[i]))
                for i in range(len(coeff))]
        idx = np.array(idxs)
    idx = idx.reshape(reshape_dims)
    return idx.astype("float32")
"""
end initializers and Theano functions
"""

"""
start training utilities
"""


def save_checkpoint(checkpoint_save_path, saver, sess):
    script_name = get_script_name()[:-3]
    save_dir = get_resource_dir(script_name)
    checkpoint_save_path = os.path.join(save_dir, checkpoint_save_path)
    saver.save(sess, checkpoint_save_path)
    logger.info("Model saved to %s" % checkpoint_save_path)


def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    partial_path = get_resource_dir("js_plot_dependencies")
    full_path = os.path.join(partial_path, "master.zip")
    url = "http://github.com/kastnerkyle/simple_template_plotter/archive/master.zip"
    if not os.path.exists(full_path):
        # Do not download automagically...
        logger.info("Download plotter template code from %s" % url)
        logger.info("Place in %s" % full_path)
        logger.info("Currently unable to save HTML, JS template unavailable")
        raise ImportError("JS Template unavailable")

    js_path = os.path.join(partial_path, "simple_template_plotter-master")
    template_path = os.path.join(js_path, "template.html")
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
    try:
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
    except ImportError:
        # The js template library wasn't found
        # Need to dump the log ...
        tmp = copy.copy(string_f)
        tmp.seek(0)
        log_output = tmp.readlines()
        del tmp
        ext = save_path.split(".")[-1]
        if ext != ".log":
            s = save_path.split(".")[:-1]
            save_path = ".".join(s + ["log"])
        with open(save_path, "w") as f:
            f.writelines(log_output)
        pass


@coroutine
def threaded_html_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, results_dict)

    Kind of overkill for an HTML writer but useful for blocking writes
    due to NFS
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


def get_script_name():
    script_path = os.path.abspath(sys.argv[0])
    # Assume it ends with .py ...
    script_name = script_path.split(os.sep)[-1]
    return script_name


def get_resource_dir(name, resource_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not resource_dir:
        resource_dir = os.getenv("TFKDLLIB_DIR", os.path.join(
            os.path.expanduser("~"), "tfkdllib_dir"))
    if folder is None:
        resource_dir = os.path.join(resource_dir, name)
    else:
        resource_dir = os.path.join(resource_dir, folder)
    if create_dir:
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
    return resource_dir


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


def run_loop(loop_function, train_itr, valid_itr, n_epochs,
             checkpoint_delay=10, checkpoint_every_n_epochs=1,
             checkpoint_every_n_updates=np.inf,
             checkpoint_every_n_seconds=1800,
             monitor_frequency=1000, skip_minimums=False,
             skip_intermediates=True, skip_most_recents=False,
             skip_n_train_minibatches=-1):
    """
    loop function must have the following api
    _loop(itr, sess, inits=None, do_updates=True)
        return cost, init_1, init_2, ....
    must pass back a list!!! For only output cost, do
        return [cost]
    do_updates will control what happens in a validation loop
    inits will pass init_1, init_2,  ... back into the loop
    TODO: add upload fields to the template, to add data to an html
    and save a copy
    loop function should return a list of [cost] + all_init_hiddens or other
    states
    """
    logger.info("Running loops...")
    _loop = loop_function
    ident = str(uuid.uuid4())[:8]
    random_state = np.random.RandomState(2177)
    monitor_prob = 1. / monitor_frequency

    checkpoint_dict = {}
    if False:
        raise ValueError("CONTINUE?")
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

    thw = threaded_html_writer()

    best_train_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    best_valid_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    # If there are more than 1M minibatches per epoch this will break!
    # Not reallocating buffer greatly helps fast training models though
    # Also we have bigger problems if there are 1M minibatches per epoch...
    # This will get sliced down to the correct number of minibatches
    # During calculations down below
    train_costs = [0.] * 1000000
    valid_costs = [0.] * 1000000
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print_network(tf.trainable_variables())
        av = tf.all_variables()
        train_saver = tf.train.Saver(av)
        valid_saver = tf.train.Saver(av)
        force_saver = tf.train.Saver(av)
        """
        Session restore?
        if os.path.exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        """
        try:
            for e in range(start_epoch, start_epoch + n_epochs):
                joint_start = time.time()
                epoch_start = time.time()
                logger.info(" ")
                logger.info("Starting training, epoch %i" % e)
                logger.info(" ")
                train_mb_count = 0
                valid_mb_count = 0
                results_dict = {k: v for k, v in checkpoint_dict.items()}
                this_results_dict = results_dict
                try:
                    train_start = time.time()
                    last_time_checkpoint = train_start
                    inits = None
                    train_itr.reset()
                    while True:
                        if train_mb_count < skip_n_train_minibatches:
                            next(train_itr)
                            train_mb_count += 1
                            continue
                        r = _loop(train_itr, sess, inits=inits, do_updates=True)
                        partial_train_costs = r[0]
                        if len(r) > 1:
                            inits = r[1:]
                        else:
                            pass
                        train_costs[train_mb_count] = np.mean(partial_train_costs)
                        tc = train_costs[train_mb_count]
                        train_mb_count += 1
                        if np.isnan(tc):
                            logger.info("NaN detected in train cost, update %i" % train_mb_count)
                            thw.close()
                            raise ValueError("NaN detected in train")

                        if (train_mb_count % checkpoint_every_n_updates) == 0:
                            checkpoint_save_path = "%s_model_update_checkpoint_%i.ckpt" % (ident, train_mb_count)
                            save_checkpoint(checkpoint_save_path, train_saver, sess)

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
                            results_save_path = "%s_model_update_results_%i.html" % (ident, train_mb_count)
                            thw.send((results_save_path, this_results_dict))
                        elif (time.time() - last_time_checkpoint) >= checkpoint_every_n_seconds:
                            time_diff = time.time() - train_start
                            last_time_checkpoint = time.time()
                            checkpoint_save_path = "%s_model_time_checkpoint_%i.ckpt" % (ident, int(time_diff))
                            #tcw.send((checkpoint_save_path, saver, sess))
                            save_checkpoint(checkpoint_save_path, train_saver, sess)

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
                            results_save_path = "%s_model_time_results_%i.html" % (ident, int(time_diff))
                            thw.send((results_save_path, this_results_dict))
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
                    # edge case - add one since stop iteration was raised
                    # before increment
                    train_costs_slice = train_costs[:train_mb_count + 1]
                    logger.info(" ")
                    logger.info("Starting validation, epoch %i" % e)
                    logger.info(" ")
                    valid_start = time.time()
                    inits = None
                    valid_itr.reset()
                    try:
                        while True:
                            r = _loop(valid_itr, sess, inits=inits, do_updates=False)
                            partial_valid_costs = r[0]
                            if len(r) > 1:
                                inits = r[1:]
                            else:
                                pass
                            valid_costs[valid_mb_count] = np.mean(partial_valid_costs)
                            vc = valid_costs[valid_mb_count]
                            valid_mb_count += 1
                            if np.isnan(vc):
                                logger.info("NaN detected in valid cost, minibatch %i" % valid_mb_count)
                                thw.close()
                                raise ValueError("NaN detected in valid")
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
                    # edge case - add one since stop iteration was raised
                    # before increment
                    valid_costs_slice = valid_costs[:valid_mb_count + 1]

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

                    mean_epoch_train_cost = np.mean(train_costs_slice)
                    # np.inf trick to avoid taking the min of length 0 list
                    old_min_train_cost = min(overall_train_costs + [np.inf])
                    if np.isnan(mean_epoch_train_cost):
                        logger.info("Previous train costs %s" % overall_train_costs[-5:])
                        logger.info("NaN detected in train cost, epoch %i" % e)
                        thw.close()
                        raise ValueError("NaN detected in train")
                    overall_train_costs.append(mean_epoch_train_cost)

                    mean_epoch_valid_cost = np.mean(valid_costs_slice)
                    old_min_valid_cost = min(overall_valid_costs + [np.inf])
                    if np.isnan(mean_epoch_valid_cost):
                        logger.info("Previous valid costs %s" % overall_valid_costs[-5:])
                        logger.info("NaN detected in valid cost, epoch %i" % e)
                        thw.close()
                        raise ValueError("NaN detected in valid")
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

                    results_dict = {k: v for k, v in checkpoint_dict.items()}

                    # Checkpointing part
                    checkpoint_start = time.time()
                    if e < checkpoint_delay or skip_minimums:
                        pass
                    elif mean_epoch_valid_cost < old_min_valid_cost:
                        logger.info("Checkpointing valid...")
                        checkpoint_save_path = "%s_model_checkpoint_valid_%i.ckpt" % (ident, e)
                        save_checkpoint(checkpoint_save_path, valid_saver, sess)
                        # tcw.send((checkpoint_save_path, saver, sess))
                        results_save_path = "%s_model_results_valid_%i.html" % (ident, e)
                        thw.send((results_save_path, results_dict))
                        logger.info("Valid checkpointing complete.")
                    elif mean_epoch_train_cost < old_min_train_cost:
                        logger.info("Checkpointing train...")
                        checkpoint_save_path = "%s_model_checkpoint_train_%i.ckpt" % (ident, e)
                        save_checkpoint(checkpoint_save_path, train_saver, sess)
                        # tcw.send((checkpoint_save_path, saver, sess))
                        results_save_path = "%s_model_results_train_%i.html" % (ident, e)
                        thw.send((results_save_path, results_dict))
                        logger.info("Train checkpointing complete.")

                    if e < checkpoint_delay:
                        pass
                        # Don't skip force checkpoints after default delay
                        # Printing already happens above
                    elif((e % checkpoint_every_n_epochs) == 0) or (e == (n_epochs - 1)):
                        logger.info("Checkpointing force...")
                        checkpoint_save_path = "%s_model_checkpoint_%i.ckpt" % (ident, e)
                        save_checkpoint(checkpoint_save_path, force_saver, sess)
                        # tcw.send((checkpoint_save_path, saver, sess))
                        results_save_path = "%s_model_results_%i.html" % (ident, e)
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
        logger.info("Unhandled minimum saving... skipped!")
        """
        # Finalize saving best train and valid
        ee = best_valid_checkpoint_epoch
        best_valid_checkpoint_dict = pickle.loads(best_valid_checkpoint_pickle)
        checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, ee)
        weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, ee)
        tcw.send((checkpoint_save_path, best_valid_checkpoint_dict))
        tww.send((weights_save_path, best_valid_checkpoint_dict))
        best_valid_results_dict = {k: v for k, v in best_valid_checkpoint_dict.items()}
        results_save_path = "%s_model_results_valid_%i.html" % (ident, ee)
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
        """
    logger.info("Loop finished, closing write threads (this may take a while!)")
    thw.close()
"""
end training utilities
"""
