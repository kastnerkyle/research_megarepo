# Author: Kyle Kastner
# License: BSD 3-Clause
# A nice overview here
# http://note.sonots.com/SciSoftware/NcutImageSegmentation.html
# See
# http://www.mathworks.com/matlabcentral/fileexchange/41526-gray-scale-image-segmentation-using-normalized-graphcuts/
# for a similar implementation in MATLAB

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.misc import imresize

"""
# Downsized lena
# Converted with imagemagick
# convert -geometry 80x80 lenna.jpg lenna.jpg
# im = plt.imread("lenna.jpg")
"""
from scipy.misc import lena
im = lena()

"""
from scipy.misc import face
im = face(gray=True)
"""

# Any bigger and my weak laptop gets memory errors
bounds = (50, 50)
im = imresize(im, bounds, interp="bicubic")

"""
from scipy.ndimage import zoom
# zoom only works from 80 -> 50, not 512 -> 50
frac_x = bounds[0] / float(im.shape[0])
frac_y = bounds[1] / float(im.shape[1])
frac = (frac_x, frac_y)
im = zoom(im, frac, order=1)
im = im[:bounds[0], :bounds[1]]
"""

sz = np.prod(im.shape)
ind = np.arange(sz)


def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return (rows, cols)

I, J = ind2sub(im.shape, ind)
I = I + 1
J = J + 1

scaling = 255.
scaled_im = im.ravel() / float(scaling)

# float32 gives the wrong answer...
scaled_im = scaled_im.astype("float64")
sim = np.zeros((sz, sz)).astype("float64")

n_splits = 2
rad = 5
sigma_x = .3
sigma_p = .1

# Faster with broadcast tricks
# Still wasting computation - einsum might be fastest
x1 = I[None]
x2 = I[:, None]
y1 = J[None]
y2 = J[:, None]
dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
scale = np.exp(-(dist / (sigma_x ** 2)))
sim = scale
sim[np.sqrt(dist) >= rad] = 0.
del x1
del x2
del y1
del y2
del dist

p1 = scaled_im[None]
p2 = scaled_im[:, None]
pdist = (p1 - p2) ** 2
pscale = np.exp(-(pdist / (sigma_p ** 2)))

sim *= pscale

dind = np.diag_indices_from(sim)
sim[dind] = 1.

"""
# Two passes over flat array
# This is sloooooooow but matches close with matlab impl
for i in range(sz):
    print(i)
    x1 = I[i]
    y1 = J[i]
    for j in range(sz):
        if i == j:
            sim[i, j] = 1.
        else:
            x2 = I[j]
            y2 = J[j]
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if np.sqrt(dist) >= rad:
                scale = 0.
            else:
                scale = np.exp(-(dist / (sigma_x ** 2)))

            pdist = (scaled_im[i] - scaled_im[j]) ** 2
            pscale = np.exp(-(pdist / (sigma_p ** 2)))
            sim[i, j] = scale * pscale
"""

d = np.sum(sim, axis=1)
D = np.diag(d)
A = (D - sim)
N = A.shape[0]

# Want second smallest eigenvector onward
S, V = eigh(A, D, eigvals=(1, n_splits + 1),
            overwrite_a=True, overwrite_b=True)
sort_ind = np.argsort(S)
S = S[sort_ind]
V = V[:, sort_ind]
segs = V
segs[:, -1] = ind


def cut(im, matches, ix, split_type="mean"):
    # Can choose how to split
    if split_type == "mean":
        split = np.mean(segs[:, ix])
    elif split_type == "median":
        split = np.median(segs[:, ix])
    elif split_type == "zero":
        split = 0.
    else:
        raise ValueError("Unknown split type %s" % split_type)

    meets = np.where(matches[:, ix] >= split)[0]
    match1 = matches[meets, :]
    res1 = np.zeros_like(im)
    match_inds = match1[:, -1].astype("int32")
    res1.ravel()[match_inds] = im.ravel()[match_inds]

    meets = np.where(matches[:, ix] < split)[0]
    match2 = matches[meets, :]
    res2 = np.zeros_like(im)
    match_inds = match2[:, -1].astype("int32")
    res2.ravel()[match_inds] = im.ravel()[match_inds]
    return (match1, match2), (res1, res2)

# Recursively split partitions
# Currently also stores intermediates
all_splits = []
all_matches = [[segs]]
for i in range(n_splits):
    matched = all_matches[-1]
    current_splits = []
    current_matches = []
    for s in matched:
        matches, splits = cut(im, s, i)
        current_splits.extend(splits)
        current_matches.extend(matches)
    all_splits.append(current_splits)
    all_matches.append(current_matches)

to_plot = all_splits[-1]
f, axarr = plt.subplots(2, len(to_plot) // 2)
for n in range(len(to_plot)):
    axarr.ravel()[n].imshow(to_plot[n], cmap="gray")
    axarr.ravel()[n].set_xticks([])
    axarr.ravel()[n].set_yticks([])
plt.show()
