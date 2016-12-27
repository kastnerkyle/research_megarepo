# Author: Kyle Kastner
# License: BSD 3-clause
# Based on example blog from
# http://kawahara.ca/matlab-jaccard-similarity-coefficient-between-images/
import numpy as np
import matplotlib.pyplot as plt


def _check_01(arr):
    """
    Check that a numpy array only has 0 and 1 in it so
    boolean operators work correctly
    """
    assert set(np.unique(arr)) == set([0, 1])

Alice = np.array([[0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0]])

RobotBob = np.array([[0, 0, 0],
                     [0, 1, 1],
                     [0, 0, 1]])

Carol = np.array([[0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0]])
_check_01(Alice)
_check_01(RobotBob)
_check_01(Carol)

f, axarr = plt.subplots(3)
plt.suptitle("3 lines comparison")
axarr[0].matshow(Alice, cmap="gray")
axarr[1].matshow(RobotBob, cmap="gray")
axarr[2].matshow(Carol, cmap="gray")

intersect = Alice & RobotBob
union = Alice | RobotBob
f, axarr = plt.subplots(2)
plt.suptitle("Intersection and union")
axarr[0].matshow(intersect, cmap="gray")
axarr[1].matshow(union, cmap="gray")

num = np.sum(intersect)
den = np.sum(union)
jaccard_index = num / float(den)
jaccard_distance = 1 - jaccard_index
print("Jaccard index %f" % jaccard_index)
print("Jaccard_distance %f" % jaccard_distance)


def calculate_jaccard_index(arr1, arr2):
    _check_01(arr1)
    _check_01(arr2)
    # This code has an edge case at 0/0 - hence the checks! 
    # You may need to manually add the 0/0 case
    intersect = arr1 & arr2
    union = arr1 | arr2
    n = np.sum(intersect)
    d = np.sum(union)
    return n / float(d)

print("Alice and RobotBob index %f" % calculate_jaccard_index(Alice, RobotBob))
print("Alice and Carol index %f" % calculate_jaccard_index(Alice, Carol))
plt.show()