# (C) Kyle Kastner, June 2014
# License: BSD 3 clause

#This code is for fun only! Use scipy.linalg.hadamard
import numpy as np
import matplotlib.pyplot as plt
import functools


def memoize(obj):
    """Memoization decorator from PythonDecoratorLibrary. Ignores
    **kwargs"""
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


@memoize
def hadamard(n):
    if n == 2:
        return np.array([[1, 1], [1, -1]], dtype='float32')
    elif (n % 2) != 0:
        raise ValueError('Value of n = %s invalid! n must be a power of 2!'%n)
    elif n < 1:
        raise ValueError('Value of n = %s invalid! n must be >= 2!'%n)
    return 2. ** (n/2) * np.kron(hadamard(2), hadamard(n/2))

def normalize(a):
    return (a - a.ravel().min()) / (a.ravel().max() - a.ravel().min())

#Generate basis functions and calculate statistics
n = 8
hvar = np.zeros((n, n))
vvar = np.zeros((n, n))
all_X = {}
for i in range(n):
    for j in range(n):
        Y = np.zeros((n, n))
        Y[i, j] = 1
        C = hadamard(n)
        X = np.dot(C.T, np.dot(Y, C))
        try:
            all_X[i][j] = X
        except KeyError:
            all_X[i] = {}
            all_X[i][j] = X

        Xfft = np.abs(np.fft.fft2(X))
        h = np.nonzero(Xfft)[1]
        print i,j
        print "h:",h
        h = h[0]
        v = np.nonzero(Xfft)[0]
        print "v:",v
        v = v[0]
        hvar[i, j] = h
        vvar[i, j] = v
hs = np.argsort(hvar, axis=1)
vs = np.argsort(vvar, axis=0)

#This gets hvar and vvar back..
vsort = hvar[vs, range(hvar.shape[0])]
hsort = vvar[range(vvar.shape[0]), hs].T
print hvar + vvar
print hsort + vsort
#print hsort.shape
#print hsort
#print vsort.shape
#print vsort
plt.figure()
plt.pcolor(hsort, cmap="gray")
plt.figure()
plt.pcolor(vsort, cmap="gray")

#plt.figure()
#plt.pcolor(hvar + vvar, cmap="gray_r")
#print hvar + vvar

f, axarr = plt.subplots(n, n)
for i in range(n):
    for j in range(n):
        idx = i
        idy = j
        X = all_X[i][j]
        axarr[idx, idy].pcolor(X, cmap='gray_r')
        axarr[idx, idy].set_frame_on(False)
        axarr[idx, idy].axes.get_xaxis().set_visible(False)
        axarr[idx, idy].axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()
