# License: GPL
# Based on scikit-tensor by mnick
# https://github.com/mnick/scikit-tensor

"""Tensor factorization."""
import numpy as np
from scipy import linalg


def _matricize(X, axis):
    dims = len(X.shape) - 1
    # If negative axis is passed, convert to equivalent positive form
    if axis < 0:
        axis = dims + axis + 1
    index = dims - axis
    return np.rollaxis(X, index).reshape(X.shape[index], -1)


def _sign_flip(X):
    max_abs_cols = np.argmax(np.abs(X), axis=0)
    signs = np.sign(X[max_abs_cols, list(range(X.shape[1]))])
    return signs * X


def _gram(X):
    return np.dot(X.T, X)


def _hosvd_init_op(X, n_components, n):
    XXT = _matricize(X, n).dot(_matricize(X, n).T)
    _, U = linalg.eigh(XXT, eigvals=(XXT.shape[0] - n_components,
                                     XXT.shape[0] - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = U[:, ::-1]
    # flip sign
    U = _sign_flip(U)
    return U


def _hosvd_init(X, n_components):
    return [_hosvd_init_op(X, n_components, i)
            for i in range(len(X.shape))][::-1]


def _random_init(X, n_components, random_state=None):
    rs = np.random.RandomState(random_state)
    re = [None] * X.ndim
    for n in range(1, X.ndim):
        re[n] = rs.rand(X.shape[n], n_components)
    return re


def _uttkrp(X, U, n):
    order = list(range(n)) + list(range(n + 1, X.ndim))
    Z = _khatrirao(tuple(U[i] for i in order), reverse=True)
    return _unfold(X, n).dot(Z)


def _khatrirao(A, reverse=False):
    if not isinstance(A, tuple):
        raise ValueError('A must be a tuple of array likes')
    N = A[0].shape[1]
    M = 1
    for i in range(len(A)):
        if A[i].ndim != 2:
            raise ValueError('A must be a tuple of matrices (A[%d].ndim = %d)' % (i, A[i].ndim))
        elif N != A[i].shape[1]:
            raise ValueError('All matrices must have same number of columns')
        M *= A[i].shape[0]
        matorder = np.arange(len(A))
        if reverse:
            matorder = matorder[::-1]
    P = np.zeros((M, N), dtype=A[0].dtype)
    for n in range(N):
        ab = A[matorder[0]][:, n]
        for j in range(1, len(matorder)):
            ab = np.kron(ab, A[matorder[j]][:, n])
            P[:, n] = ab
    return P


def _from_to_without(frm, to, without, step=1, skip=1, reverse=False, separate=False):
    if reverse:
        frm, to = (to - 1), (frm - 1)
        step *= -1
        skip *= -1
    a = list(range(frm, without, step))
    b = list(range(without + skip, to, step))
    if separate:
        return a, b
    else:
        return a + b


def _unfold(X, mode):
    sz = np.array(X.shape)
    N = len(sz)
    order = ([mode], _from_to_without(N - 1, -1, mode, step=-1, skip=-1))
    newsz = (sz[order[0]], np.prod(sz[order[1]]))
    arr = X.transpose((order[0] + order[1]))
    arr = arr.reshape(newsz)
    return arr


def _magic_cpN(X, n_components, tol, max_iter, init_type, random_state):
    N = X.ndim
    if init_type == "random":
        U = _random_init(X, n_components, random_state)
    else:
        U = _hosvd_init(X, n_components)
    for itr in range(max_iter):
        for n in range(N):
            Unew = _uttkrp(X, U, n)
            Y = np.ones((n_components, n_components))
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * np.dot(U[i].T, U[i])
            Unew = Unew.dot(linalg.pinv(Y))
            # Normalize
            if itr == 0:
                lmbda = np.sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda
    return U


def cp(X, n_components=None, tol=1E-5, max_iter=500, init_type="hosvd",
       random_state=None, force_general=True):
    if n_components is None:
        raise ValueError("n_components is a required argument!")

    return _magic_cpN(X, n_components, tol=tol, max_iter=max_iter,
                      init_type=init_type, random_state=random_state)

if __name__ == "__main__":
    from scipy.io.matlab import loadmat
    import matplotlib.pyplot as plt
    # You can download this data set here:
    # http://www.models.life.ku.dk/Sensory_Bread
    mat = loadmat('brod.mat')
    X = mat['X'].reshape(mat['DimX'].ravel())
    X_flat = X.reshape(X.shape[0], -1)
    U0, U1, U2 = cp(X, n_components=2, init_type='hosvd', random_state=1999)
    t = np.dot(X_flat.T, U0)
    plt.scatter(t[:, 0], t[:, 1])
    plt.show()