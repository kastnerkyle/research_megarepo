# License: BSD 3-clause
# Authors: Kyle Kastner,
#          Pierre Luc Carrier

import numpy as np
import scipy
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy import linalg, fftpack


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    :usage:
        >>> librosa.mel_to_hz(3)
        array([ 200.])

        >>> librosa.mel_to_hz([1,2,3,4,5])
        array([  66.66666667,  133.33333333,  200.        ,  266.66666667,
                333.33333333])

    :parameters:
      - mels          : np.ndarray [shape=(n,)], float
          mel bins to convert
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - frequencies   : np.ndarray [shape=(n,)]
          input mels in Hz
    """

    mels = np.asarray([mels], dtype=float).flatten()

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :usage:
        >>> librosa.hz_to_mel(60)
        array([0.9])
        >>> librosa.hz_to_mel([110, 220, 440])
        array([ 1.65,  3.3 ,  6.6 ])

    :parameters:
      - frequencies   : np.ndarray [shape=(n,)] , float
          scalar or array of frequencies
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - mels        : np.ndarray [shape=(n,)]
          input frequencies in Mels
    """

    frequencies = np.asarray([frequencies]).flatten()

    if np.isscalar(frequencies):
        frequencies = np.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False,
                    extra=False):
    """Compute the center frequencies of mel bands

    :usage:
        >>> librosa.mel_frequencies(n_mels=40)
        array([    0.        ,    81.15543818,   162.31087636,   243.46631454,
                324.62175272,   405.7771909 ,   486.93262907,   568.08806725,
                649.24350543,   730.39894361,   811.55438179,   892.70981997,
                973.86525815,  1058.38224675,  1150.77458676,  1251.23239132,
                1360.45974173,  1479.22218262,  1608.3520875 ,  1748.75449257,
                1901.4134399 ,  2067.39887435,  2247.87414245,  2444.10414603,
                2657.46420754,  2889.44970936,  3141.68657445,  3415.94266206,
                3714.14015814,  4038.36904745,  4390.90176166,  4774.2091062 ,
                5190.97757748,  5644.12819182,  6136.83695801,  6672.55713712,
                7255.04344548,  7888.37837041,  8577.0007833 ,  9325.73705043])

    :parameters:
      - n_mels    : int > 0 [scalar]
          number of Mel bins

      - fmin      : float >= 0 [scalar]
          minimum frequency (Hz)

      - fmax      : float >= 0 [scalar]
          maximum frequency (Hz)

      - htk       : bool
          use HTK formula instead of Slaney

      - extra     : bool
          include extra frequencies necessary for building Mel filters

    :returns:
      - bin_frequencies : ndarray [shape=(n_mels,)]
          vector of Mel frequencies
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz_to_mel(fmin, htk=htk)
    maxmel = hz_to_mel(fmax, htk=htk)

    mels = np.arange(minmel, maxmel + 1, (maxmel - minmel) / (n_mels + 1.0))

    if not extra:
        mels = mels[:n_mels]

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of ``np.fft.fftfreqs``

    :usage:
        >>> librosa.fft_frequencies(sr=22050, n_fft=16)
        array([     0.   ,   1378.125,   2756.25 ,   4134.375,   5512.5  ,
                 6890.625,   8268.75 ,   9646.875,  11025.   ])

    :parameters:
      - sr : int > 0 [scalar]
          Audio sampling rate

      - n_fft : int > 0 [scalar]
          FFT window size

    :returns:
      - freqs : np.ndarray [shape=(1 + n_fft/2,)]
          Frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft/2),
                       endpoint=True)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :usage:
        >>> mel_fb = librosa.filters.mel(22050, 2048)

        >>> # Or clip the maximum frequency to 8KHz
        >>> mel_fb = librosa.filters.mel(22050, 2048, fmax=8000)

    :parameters:
      - sr        : int > 0 [scalar]
          sampling rate of the incoming signal

      - n_fft     : int > 0 [scalar]
          number of FFT components

      - n_mels    : int > 0 [scalar]
          number of Mel bands to generate

      - fmin      : float >= 0 [scalar]
          lowest frequency (in Hz)

      - fmax      : float >= 0 [scalar]
          highest frequency (in Hz).
          If ``None``, use ``fmax = sr / 2.0``

      - htk       : bool [scalar]
          use HTK formula instead of Slaney

    :returns:
      - M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
          Mel transform matrix
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft / 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = mel_frequencies(n_mels,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk,
                            extra=True)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - fftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


def melspec(S, fs, nmelbands=128):
    nfft = 2 * (S.shape[1] - 1)
    melfb = mel(fs, nfft)
    return melfb.dot(S.T).T


def invmelspec(M, fs, nfft=1024):
    melfb = mel(fs, nfft, n_mels=M.shape[1])
    ww = np.dot(melfb.T, melfb)
    iwts = (melfb / np.maximum(np.mean(np.diag(ww)) / 100,
                               np.sum(ww, axis=1))).T
    return np.dot(iwts, M.T).T


def stft(x, fftsize=256, overlap=2):
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+fftsize])
                     for i in range(0, len(x)-fftsize, hop)])


def istft(X, overlap=2):
    fftsize=(X.shape[1] - 1) * 2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize + 1)[:-1]
    x = scipy.zeros(X.shape[0] * hop)
    wsum = scipy.zeros(X.shape[0] * hop)
    for n,i in enumerate(range(0, len(x) - fftsize, hop)):
        x[i:i + fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i + fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def voiced_unvoiced(X, window_size=257, window_step=128, copy=True):
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
    voiced_unvoiced = np.zeros((n_windows, 2))
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
        start = np.max([20, start[0]])
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

    Parameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approximately signals
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
    X *= 2 ** 15
    # 90% volume to try and avoid clipping
    X *= .9
    return X.astype('int16')


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
        Residual excitations. If None, this will be synthesized with white noise.

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
            # No voiced/unvoiced info, so just use randn
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


def sinusoid_analysis(X, input_sample_rate, resample_block=128, copy=True):
    """
    Contruct a sinusoidal model for the input signal.

    Parameters
    ----------
    X : ndarray
        Input signal to model

    input_sample_rate : int
        The sample rate of the input signal

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    frequencies_hz : ndarray
       Frequencies for the sinusoids, in Hz.

    magnitudes : ndarray
       Magnitudes of sinusoids returned in ``frequencies``

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    resample_to = 8000
    if input_sample_rate != resample_to:
        if input_sample_rate % resample_to != 0:
            raise ValueError("Input sample rate must be a multiple of 8k!")
        # Should be able to use resample... ?
        # resampled_count = round(len(X) * resample_to / input_sample_rate)
        # X = sg.resample(X, resampled_count, window=sg.hanning(len(X)))
        X = sg.decimate(X, input_sample_rate // resample_to)
    step_size = 2 * int(round(resample_block / float(input_sample_rate) * resample_to / 2.))
    a, g, e = lpc_analysis(X, order=8, window_step=step_size,
                           window_size=2 * step_size)
    f, m = lpc_to_frequency(a, g)
    f_hz = f * resample_to / (2 * np.pi)
    return f_hz, m


def slinterp(X, factor, copy=True):
    """
    Slow-ish linear interpolation of a 1D numpy array. There must be some
    better function to do this in numpy.

    Parameters
    ----------
    X : ndarray
        1D input array to interpolate

    factor : int
        Integer factor to interpolate by

    Return
    ------
    X_r : ndarray
    """
    sz = np.product(X.shape)
    X = np.array(X, copy=copy)
    X_s = np.hstack((X[1:], [0]))
    X_r = np.zeros((factor, sz))
    for i in range(factor):
        X_r[i, :] = (factor - i) / float(factor) * X + (i / float(factor)) * X_s
    return X_r.T.ravel()[:(sz - 1) * factor + 1]


def sinusoid_synthesis(frequencies_hz, magnitudes, input_sample_rate=16000,
                       resample_block=128):
    """
    Create a time series based on input frequencies and magnitudes.

    Parameters
    ----------
    frequencies_hz : ndarray
        Input signal to model

    magnitudes : int
        The sample rate of the input signal

    input_sample_rate : int, optional (default=16000)
        The sample rate parameter that the sinusoid analysis was run with

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    rows, cols = frequencies_hz.shape
    synthesized = np.zeros((1 + ((rows - 1) * resample_block),))
    for col in range(cols):
        mags = slinterp(magnitudes[:, col], resample_block)
        freqs = slinterp(frequencies_hz[:, col], resample_block)
        cycles = np.cumsum(2 * np.pi * freqs / float(input_sample_rate))
        sines = mags * np.cos(cycles)
        synthesized += sines
    return synthesized


def compress(X, n_components, window_size=128):
    """
    Compress using the DCT

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        The input signal to compress. Should be 1-dimensional

    n_components : int
        The number of DCT components to keep. Setting n_components to about
        .5 * window_size can give compression with fairly good reconstruction.

    window_size : int
        The input X is broken into windows of window_size, each of which are
        then compressed with the DCT.

    Returns
    -------
    X_compressed : ndarray, shape=(num_windows, window_size)
       A 2D array of non-overlapping DCT coefficients. For use with uncompress

    Reference
    ---------
    http://nbviewer.ipython.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb
    """
    if len(X) % window_size != 0:
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))
    num_frames = len(X) // window_size
    X_strided = X.reshape((num_frames, window_size))
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


def uncompress(X_compressed, window_size=128):
    """
    Uncompress a DCT compressed signal (such as returned by ``compress``).

    Parameters
    ----------
    X_compressed : ndarray, shape=(n_samples, n_features)
        Windowed and compressed array.

    window_size : int, optional (default=128)
        Size of the window used when ``compress`` was called.

    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples)
        Reconstructed version of X.
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0], window_size - X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return X_r.ravel()


def apply_kaiserbessel_window(X, alpha=6.5):
    """
    Apply a Kaiser-Bessel window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    alpha : float, optional (default=6.5)
        Tuning parameter for Kaiser-Bessel function. alpha=6.5 should make
        perfect reconstruction possible for MDCT.

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    beta = np.pi * alpha
    win = sg.kaiser(X.shape[1], beta)
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


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
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),))
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


if __name__ == "__main__":
    # ae.wav is from
    # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav
    # Partially following the formant tutorial here
    # http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
    from scipy.io import wavfile
    from sklearn.cluster import KMeans
    samplerate, X = wavfile.read('ae.wav')
    # MATLAB wavread does normalization
    X = X.astype('float64') / (2 ** 15)

    print("Calculating sinusoids")
    f_hz, m = sinusoid_analysis(X, input_sample_rate=16000)
    sig = np.concatenate((f_hz, m), axis=1)
    tf = KMeans(n_clusters=10, random_state=1999)
    tf.fit(sig)

    from IPython import embed; embed()
    raise ValueError()

    Xs_sine = sinusoid_synthesis(f_hz, m)
