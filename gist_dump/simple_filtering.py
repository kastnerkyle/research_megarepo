# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

sine = np.sin(np.linspace(-4 * np.pi, 4 * np.pi, 10000))
noise = 0.2 * np.random.randn(len(sine))
s = sine + noise

plt.plot(s)
plt.figure()

low_pass = [1, 1] * 128
plt.plot(lfilter(low_pass, [1], s))
plt.figure()

high_pass = [1, -1] * 128
plt.plot(lfilter(high_pass, [1], s))
plt.show()