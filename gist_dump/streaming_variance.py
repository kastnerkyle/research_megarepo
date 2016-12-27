# (C) Kyle Kastner, June 2014
# License: BSD 3 clause

import numpy as np

# Using data from http://www.mathsisfun.com/data/standard-deviation.html
X = np.array([600, 470, 170, 430, 300])

# Showing steps from basic to Welford's and batch
# See http://cpsc.yale.edu/sites/default/files/files/tr222.pdf
# Simple formula
print("Simple calculation")
batch_var = np.sum((X - np.mean(X)) ** 2) / X.shape[0]
print(batch_var)

print("One-pass naive algorithm")
batch_var = (np.sum(X ** 2) - 1 / X.shape[0] * (np.sum(X)) ** 2) / X.shape[0]
print(batch_var)

print("Welford's")
# From John D. Cook http://www.johndcook.com/standard_deviation.html
batch_mean = X[0]
batch_var = 0
n_samples = len(X)
for j in range(1, len(X[1:]) + 1):
    old_mean = batch_mean
    old_var = batch_var
    batch_mean = old_mean + 1. / (j + 1) * (X[j] - old_mean)
    batch_var = old_var + (X[j] - old_mean) * (X[j] - batch_mean)
batch_var /= n_samples
print(batch_var)

print("Youngs and Cramer")
# From http://cpsc.yale.edu/sites/default/files/files/tr222.pdf
batch_sum = X[0]
batch_var = 0
n_samples = len(X)
for j in range(1, len(X[1:]) + 1):
    batch_sum = batch_sum + X[j]
    batch_var = batch_var + 1. / ((j + 1) * j) * ((j + 1) * X[j] - batch_sum) ** 2
batch_var /= n_samples
print(batch_var)

print("Batch Youngs and Cramer")
# From http://cpsc.yale.edu/sites/default/files/files/tr222.pdf
idx = 2
m = len(X[:idx])
n = len(X[idx:])
n_samples = len(X)
b1_sum = np.sum(X[:idx])
b1_var = np.sum((X[:idx] - 1. / m * b1_sum) ** 2)
b2_sum = np.sum(X[idx:])
b2_var = np.sum((X[idx:] - 1. / n * b2_sum) ** 2)
batch_sum = b1_sum + b2_sum
partial_var = float(m) / (n * (m + n)) * (n / float(m) *
                                          b1_sum - b2_sum) ** 2
batch_var = b1_var + b2_var + partial_var
batch_var /= n_samples
print(batch_var)