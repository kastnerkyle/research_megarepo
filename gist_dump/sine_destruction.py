# Author: Kyle Kastner
# License: BSD 3-clause
import matplotlib.pyplot as plt
import numpy as np


fs = 100  # sample rate of 100 samples / sec, with max f 50
f = 5  # 5 Hz frequency
samples = 25  # .25 seconds of samples @ 100 samples / sec
x = np.arange(samples)
y1 = np.sin(2 * np.pi * f * x / fs + .5 * np.pi)
y2 = np.sin(2 * np.pi * f * x / fs + -.5 * np.pi)
plt.plot(y1)
plt.plot(y2)
plt.plot(y1 + y2)
plt.show()