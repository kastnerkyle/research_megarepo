import numpy as np
from numpy.testing import assert_almost_equal

A = np.arange(200).reshape(40, 5).astype('float32')
B = np.arange(100).reshape(20, 5) + 10.
C = np.vstack((A, B))

mA = np.mean(A, axis=0)
mB = np.mean(B, axis=0)
mC = np.mean(C, axis=0)

nA = len(A)
nB = len(B)
nC = len(C)

# Outer product for mean subtraction works
pre_meanA = np.dot((A - mA).T, (A - mA))
post_meanA = np.dot(A.T, A) - np.dot(mA[None].T, mA[None]) * nA
pre_meanB = np.dot((B - mB).T, (B - mB))
post_meanB = np.dot(B.T, B) - np.dot(mB[None].T, mB[None]) * nB
pre_meanC = np.dot((C - mC).T, (C - mC))
post_meanC = np.dot(C.T, C) - np.dot(mC[None].T, mC[None]) * nC

assert_almost_equal(pre_meanA, post_meanA)
assert_almost_equal(pre_meanB, post_meanB)
assert_almost_equal(pre_meanC, post_meanC)

cA = np.dot(A.T, A)
cB = np.dot(B.T, B)
cC = np.dot(C.T, C)

# Combine means
joined_mean = (mA * nA + mB * nB)/nC
joined_C = cA + cB
# Equivalence
assert_almost_equal(cA + cB, cC)
assert_almost_equal(joined_mean, mC)
joined_C = joined_C - np.dot(joined_mean[None].T, joined_mean[None]) * nC
assert_almost_equal(joined_C, post_meanC)