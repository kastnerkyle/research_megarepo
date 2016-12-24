import numpy as np

def exponential_kernel(x1, x2):
    # Broadcasting tricks to get every pairwise distance.
    return np.exp(-((x1[np.newaxis, :, :] - x2[:, np.newaxis, :]) ** 2).sum(2)).T

a = np.random.randn(100, 5)

print(exponential_kernel(a, a).shape)