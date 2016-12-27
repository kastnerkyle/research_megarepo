import numpy as np
import theano
import theano.tensor as T
import pickle

d = pickle.load(open("info.pkl", mode="rb"))

X = d["X"]
y = d["y"]
fit_function2 = d["fit_function"]
predict_function2 = d["predict_function"]

def p2():
    return predict_function2(X)

def f2():
    return fit_function2(X, y)

from IPython import embed; embed()