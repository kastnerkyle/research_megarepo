# Author: Kyle Kaster
# License: BSD 3-clause
import numpy as np


def online_stats(X):
    """
    Converted from John D. Cook
    http://www.johndcook.com/blog/standard_deviation/
    """
    prev_mean = None
    prev_var = None
    n_seen = 0
    for i in range(len(X)):
        n_seen += 1
        if prev_mean is None:
            prev_mean = X[i]
            prev_var = 0.
        else:
            curr_mean = prev_mean + (X[i] - prev_mean) / n_seen
            curr_var = prev_var + (X[i] - prev_mean) * (X[i] - curr_mean)
            prev_mean = curr_mean
            prev_var = curr_var
    # n - 1 for sample variance, but numpy default is n
    return prev_mean, np.sqrt(prev_var / n_seen)

from numpy.testing import assert_almost_equal
X = np.random.rand(10000, 50)
tm = X.mean(axis=0)
ts = X.std(axis=0)
sm, ss = online_stats(X)
assert_almost_equal(tm, sm)
assert_almost_equal(ts, ss)
