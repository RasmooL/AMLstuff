import sys
import numpy as np
import theano
import theano.tensor as T

# Recursive neural network with no hidden layer
class RNN(object):
    def __init__(self, n_in, n_out, activation=T.tanh):
        self.activation = activation

        self.x = T.matrix()

        W_init = np.asarray(np.random.uniform(  size = (n_out, n_in),
                                                low = -0.01, high = 0.01),
                                                dtype = theano.config.floatX)
        self.W = theano.shared(value = W_init, name = 'W')

        W_score_init = np.asarray(np.random.uniform(size = (n_out, n_in),
                                                    low = -0.01, high=0.01),
                                                    dtype = theano.config.floatX)
        self.W_score = theano.shared(value = W_score_init, name = 'W_score')

        b_init = np.zeros((n_out,), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_init, name = 'b')

        self.params = [self.W, self.b]

        # Recursive function
        def step(x_t):
            return self.activation(T.dot(x_t, self.W) + self.b)

        [self.p],_ = theano.scan(step, sequences = x)
        self.predict = theano.function(inputs=self.x, outputs=self.p)
        
