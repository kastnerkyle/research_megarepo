import numpy as np
import matplotlib.pyplot as plt

control = np.zeros(10000)
control[:2000] = -50
control[5000:7000] = 50.
control[7000:] = -50


def frequency_modulation(modulation_signal, carrier_freq=100.,
                         sampling_freq=44100., modulation_strength=1.):
    assert modulation_signal.ndim == 1
    t = np.arange(len(modulation_signal)) / sampling_freq
    integrated = np.cumsum(modulation_signal) / sampling_freq
    modulated = np.cos(2. * np.pi * (carrier_freq * t +
                                     modulation_strength * integrated))
    return modulated

y = frequency_modulation(control)
plt.plot(y)
plt.show()