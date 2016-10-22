import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys

fs, d = wavfile.read(sys.argv[1])
d = d - d.mean()
plt.specgram(d, NFFT=256, noverlap=128, cmap="gray", interpolation=None)
plt.title("kiwi")
plt.show()
