# Author: Kyle Kastner
# Thanks to LD for mathematical guidance
# License: BSD 3-Clause
# See pseudocode for minibatch kmeans
# https://algorithmicthoughts.wordpress.com/2013/07/26/machine-learning-mini-batch-k-means/
# Unprincipled and hacky recentering to median at the end of function
import numpy as np
from scipy.cluster.vq import vq


def minibatch_kmedians(X, M=None, n_components=10, n_iter=100,
                       minibatch_size=100, random_state=None):
    n_clusters = n_components
    if M is not None:
        assert M.shape[0] == n_components
        assert M.shape[1] == X.shape[1]
    if random_state is None:
        random_state = np.random.RandomState(random_state)
    elif not hasattr(random_state, 'shuffle'):
        # Assume integer passed
        random_state = np.random.RandomState(int(random_state))
    if M is None:
        ind = np.arange(len(X)).astype('int32')
        random_state.shuffle(ind)
        M = X[ind[:n_clusters]]

    center_counts = np.zeros(n_clusters)
    pts = list(np.arange(len(X), minibatch_size)) + [len(X)]
    if len(pts) == 1:
        # minibatch size > dataset size case
        pts = [0, None]
    minibatch_indices = zip(pts[:-1], pts[1:])
    for i in range(n_iter):
        for mb_s, mb_e in minibatch_indices:
            Xi = X[mb_s:mb_e]
            # Broadcasted Manhattan distance
            # Could be made faster with einsum perhaps
            centers = np.abs(Xi[:, None, :] - M[None]).sum(
                axis=-1).argmin(axis=1)

            def count_update(c):
                center_counts[c] += 1
            [count_update(c) for c in centers]
            scaled_lr = 1. / center_counts[centers]
            Mi = M[centers]
            scaled_lr = scaled_lr[:, None]
            # Gradient of abs
            Mi = Mi - scaled_lr * ((Xi - Mi) / np.sqrt((Xi - Mi) ** 2 + 1E-9))
            M[centers] = Mi
    # Reassign centers to nearest datapoint
    mem, _ = vq(M, X)
    M = X[mem]
    return M


if __name__ != "__main_":
    random_state = np.random.RandomState(1999)
    Xa = random_state.randn(200, 2)
    Xb = .25 * random_state.randn(200, 2) + np.array((5, 3))
    X = np.vstack((Xa, Xb))
    ind = np.arange(len(X))
    random_state.shuffle(ind)
    X = X[ind]
    M1 = minibatch_kmedians(X, n_iter=1, random_state=random_state)
    M2 = minibatch_kmedians(X, M1, n_iter=1000, random_state=random_state)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    raise ValueError()
