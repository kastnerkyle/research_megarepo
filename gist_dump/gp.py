import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.externals import joblib

mem = joblib.Memory(cachedir='.')


def plot_gp_confidence(gp, show_gp_points=True, X_low=-1, X_high=1,
                       X_count=1000, xlim=None, ylim=None, show=False):
    xpts = np.linspace(X_low, X_high, X_count).reshape((-1, 1))
    v = [gp.predict(x) for x in xpts]
    if v[0][0].shape[1] > 1:
        raise ValueError("plot_gp_confidence only works for 1D GP")
    means = np.array([i[0] for i in v]).squeeze()
    sigmas = np.array([i[1] for i in v]).squeeze()
    plt.errorbar(xpts.squeeze(), means, yerr=sigmas,
                 capsize=0, color="steelblue")

    if show_gp_points:
        plt.plot(gp._X, gp._y, color="darkred", marker="o", linestyle="")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if show:
        plt.show()


kernel_default_params = {}
kernel_default_params["exponential_kernel"] = [1, 1]


def exponential_kernel(x1, x2, params):
    assert len(params) == 2
    return params[0] * np.exp(-0.5 * params[1] * (
        x1[None, :, :] - x2[:, None, :])[:, :, 0] ** 2)


def _covariance(kernel, kernel_params, x1, x2):
    return kernel(x1, x2, kernel_params)


def _covariance_inv(kernel, kernel_params, x1, x2):
    return linalg.pinv(_covariance(kernel, kernel_params, x1, x2))

_cache_covariance_inv = mem.cache(_covariance_inv)


def check_array(arr, copy=False):
    """
    Forces 2D!
    """
    if copy:
        arr = np.copy(arr)
    if arr.ndim < 2:
        arr = arr[:, None]
    return arr


class ExperimentalGaussianProcessRegressor(object):
    def __init__(self, kernel, kernel_params=None, copy=True):
        if kernel == "exp":
            self.kernel = exponential_kernel
            if kernel_params is None:
                self.kernel_params = kernel_default_params["exponential_kernel"]
            else:
                self.kernel_params = kernel_params
        else:
            self.kernel = kernel
            if kernel_params is not None:
                self.kernel_params = kernel_params
            else:
                raise ValueError("kernel_params cannot be None "
                                 "for custom kernels!")
        self.copy = copy
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = None
        self._y = None
        self.partial_fit(X, y)

    def partial_fit(self, X, y):
        X = check_array(X, self.copy)
        y = check_array(y, self.copy)
        if self._X is None:
            self._X = X
            self._y = y
        else:
            self._X = np.vstack((self._X, X))
            self._y = np.vstack((self._y, y))

    def predict(self, X, y=None):
        X = check_array(X, copy=self.copy)
        kernel = self.kernel
        kernel_params = self.kernel_params
        if self._X is not None:
            cov_xnx = _covariance(kernel, kernel_params, X, self._X)
            cov_x_inv = _cache_covariance_inv(kernel, kernel_params, self._X,
                                              self._X)
            cov_xn = _covariance(kernel, kernel_params, X, X)
            # partial = k*T M^-1
            # based on these slides
            # https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf
            partial = cov_xnx.T.dot(cov_x_inv) #cov_x_inv.dot(cov_xnx).T
            mean = partial.dot(self._y)
            std = cov_xn - partial.dot(cov_xnx)
        else:
            # enable prediction from unconditional GP
            zeros = np.zeros_like(X)
            std = _covariance(kernel, kernel_params, zeros, zeros)
            mean = zeros
            std = np.ones_like(X) * std[None]
        return mean, std

# Data to match example from
# https://github.com/fonnesbeck/Bios8366/blob/master/notebooks/Section5_1-Gaussian-Processes.ipynb
X_pts = [1., -.7, -2.1, -1.5, 0.3, 1.8, 2.5]
y_pts = [-0.7607861506012358, 0.37195200694374364, 0.61206537, 1.19877915,
         -0.28605053, 3.24753331, 0.21642368]
X = np.array(X_pts)
y = np.array(y_pts)

gp = ExperimentalGaussianProcessRegressor("exp", kernel_params=[1, 10])
plot_gp_confidence(gp, show_gp_points=True, X_low=-3, X_high=3,
                   X_count=1000,
                   xlim=[-3, 3], ylim=[-3, 3])
plt.figure()
gp.fit(X[:2], y[:2])
plot_gp_confidence(gp, show_gp_points=True, X_low=-3, X_high=3,
                   X_count=1000,
                   xlim=[-3, 3], ylim=[-3, 3])
plt.figure()
gp.fit(X, y)
plot_gp_confidence(gp, show_gp_points=True, X_low=-3, X_high=3, X_count=1000,
                   xlim=[-3, 3], ylim=[-3, 3])
plt.show()