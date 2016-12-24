import numpy as np

# Broadcast tricks to repeat a matrix
a = np.arange(100 * 10).reshape((100, 10))
# Number of times to clone each entry
clone_count = 2
# axis 0 clone
b = np.ones((1, clone_count, a.shape[1]))
c = (a[:, None, :] * b).reshape((-1, a.shape[-1]))
# axis 1 clone
b = np.ones((clone_count, 1, a.shape[1]))
c = (a[None, :, :] * b).reshape((-1, a.shape[-1]))