import numpy
import theano
import theano.tensor as T


# number of input units
nin = 5
# number of output units
nout = 5

# input (where first dimension is time)
u = T.matrix()
# target (where first dimension is time)
t = T.matrix()
# learning rate
lr = T.scalar()
# recurrent weights as a shared variable
W = theano.shared(numpy.random.uniform(size=(nout, nin), low=-.01, high=.01))
# biases
b = theano.shared(numpy.random.uniform(size=nout, low=-.01, high=.01))


# recurrent function (using tanh activation function) and linear output
# activation function
def step(x_t, W, b):
    p = T.tanh(T.dot(x_t, W) + b)
    return p

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h,y], _ = theano.scan(step,
                    sequences=[u],
                    non_sequences=[W, b])
# # error between output and target
# error = ((y - t) ** 2).sum()
# # gradients on the weights using BPTT
# gW, gb = T.grad(error, [W, b])
# # training function, that computes the error and updates the weights using
# # SGD.
# fn = theano.function([u, t, lr],
#                      error,
#                      updates={W: W - lr * gW,
#                             b: b - lr * gb})
