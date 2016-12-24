import numpy as np
import theano
import theano.tensor as T
import pickle

class sgd_nesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates

X = np.random.randn(100, 12).astype(theano.config.floatX)
y = np.random.randint(0, 4, 100).astype("int32")
W = np.random.randn(12, 5).astype(theano.config.floatX)
b = np.zeros(5).astype(theano.config.floatX)

X_sym = T.matrix()
X_sym.tag.test_value = X
y_sym = T.lvector()
y_sym.tag.test_value = y

W_shared = theano.shared(W)
b_shared = theano.shared(b)
params = [W_shared, b_shared]

theano.printing.Print("X_sym.shape")(X_sym.shape)
out = T.dot(X_sym, W_shared) + b_shared
theano.printing.Print("out.shape")(out.shape)
probs = T.nnet.softmax(out)
preds = T.argmax(probs, axis=1)
cost = -T.mean(T.log(probs)[T.arange(y_sym.shape[0]), y_sym])
grads = T.grad(cost, params)
opt = sgd_nesterov(params)
updates = opt.updates(params, grads, 0.01, 0.8)

predict_function = theano.function([X_sym], preds)
fit_function = theano.function([X_sym, y_sym], cost, updates=updates)

pickle.dump({"X":X, "y":y, "fit_function":fit_function, "predict_function":predict_function}, open("info.pkl", mode="wb"))

def p():
    return predict_function(X)

def f():
    return fit_function(X, y)

for i in range(10):
    c = f()
    print(c)
