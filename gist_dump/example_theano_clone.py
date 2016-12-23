import theano.tensor as T
import theano

a = T.vector()
b = T.matrix()
fa = a ** 2
f = theano.function([a], fa)
f2 = theano.function([b], theano.clone(fa, replace={a: b}, strict=False))
print(f([2]))
print(f2([[2]]))