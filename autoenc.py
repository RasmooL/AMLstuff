import sys
import re
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pylab as plt
import os.path
from node import Node
from rnn import RNN
from util import *

theano.config.floatX = 'float64'

# Load word embeddings
word2vec = load_embeddings('vectors.6B.50d.txt')

# Load corpus
corpus = []
with open('corpus') as f:
    for line in f:
        line = line.lower()
        line = re.sub(r"[^a-z ]+", '', line) # Remove many characters
        sentence = line.split(' ')
        sentence = [w for w in sentence if w != ''] # Remove spaces

        corpus.append(string_embedding(word2vec, sentence))

# Get input string (question) and convert to vectors
sys.argv.pop(0) # Remove program name
input_embed = string_embedding(word2vec, sys.argv)

### Autoencoder weights
x = T.dvector(name='input')
n_in = 100
n_hidden = 50

if os.path.isfile('weights.npz'):
    npzfile = np.load('weights.npz')
    W = theano.shared(value = npzfile['W'], name = 'W')
    b = theano.shared(value = npzfile['b'], name = 'b')
    U = theano.shared(value = npzfile['U'], name = 'U')
    c = theano.shared(value = npzfile['c'], name = 'c')
else:
    lohi = np.sqrt(6. / (n_in + n_hidden))
    W_init = np.asarray(np.random.uniform(  size = (n_hidden, n_in),
                                            low = -lohi, high = lohi),
                                            dtype = theano.config.floatX)
    W = theano.shared(value = W_init, name = 'W')

    b_init = np.asarray(np.random.uniform(  size = (n_hidden),
                                            low = -lohi, high = lohi),
                                            dtype = theano.config.floatX)
    b = theano.shared(value = b_init, name = 'b')

    U_init = np.asarray(np.random.uniform(  size = (n_in, n_hidden),
                                            low = -lohi, high = lohi),
                                            dtype = theano.config.floatX)
    U = theano.shared(value = U_init, name = 'U')

    c_init = np.asarray(np.random.uniform(  size = (n_in),
                                            low = -lohi, high = lohi),
                                            dtype = theano.config.floatX)
    c = theano.shared(value = c_init, name = 'c')

params = [W, b, U, c]

# Hidden, reconstruction
h = T.tanh(T.dot(W, x) + b)
r = T.tanh(T.dot(U, h) + c)
cost = (x - r).norm(2) ** 2
get_parent = theano.function([x], [h, r, cost])

# Build tree (each recursion combines two nodes into one parent)
def build_recursive(tree):
    if not tree:
        return None # Empty tree

    # Find best new node in current recursion
    best_cost = np.inf
    best_i = None
    best_node = None
    for i in xrange(len(tree) - 1):
        first = tree[i]
        second = tree[i+1]

        # Calculate reconstruction cost for current pair
        input = np.concatenate([first.value, second.value])
        h,r,c = get_parent(input) # Feed input through autoencoder (hidden, reconstruction, cost)

        if c < best_cost:
            best_cost = c
            best_i = i
            best_node = Node(name=c, first=first, second=second, value=h) # Simple linked node

    # Now that we have the best node, replace children in tree with parent
    tree.pop(best_i)
    tree.pop(best_i)
    tree.insert(best_i, best_node)
    if len(tree) == 1:
        return tree[0] # Root node, done
    return build_recursive(tree)

# Implementation for theano.scan
def build_scan(tree):
    if not tree:
        raise Exception("Empty tree!")

    out, upd = theano.scan( fn=blah,
                            n_steps = len(tree) - 1)
    return tree, theano.scan_module.until(len(tree) == 1) # Recurse until root node is found

outputs, updates = theano.scan( fn=build_scan,
                                sequences=None,
                                outputs_info=None,
                                non_sequences=None,
                                n_steps=1000 # Maximum number of recursions (because of until)
)

# Train
gparams = T.grad(cost, params)
updates = []
for p, gp in zip(params, gparams):
    p_update = theano.shared(p.get_value()*0.)
    updates.append((p, p - 0.005 * p_update))
    updates.append((p_update, 0.8 * p_update + (1. - 0.8)*gp))
#updates = [(p, p - 0.07 * gp) for p, gp in zip(params, gparams)]
train = theano.function([x], cost, updates=updates)
get_params = theano.function([], [W, b, U, c])

# Train on corpus (1 layer)
for i in xrange(500):
    for n in xrange(len(corpus) - 1):
        sen = corpus[n]
        cost = 0
        for w in xrange(len(sen) - 1):
                input = np.concatenate([sen[w], sen[w+1]])
                cost = cost + train(input)
    if i % 49 == 0:
        mean_cost = cost / len(sen)
        print "Current mean cost:" + str(cost)
#print "Params: "
#print get_params()

# Test
sent = ['the', 'cat', 'sat', 'on', 'the', 'mat']
test_tree = make_node_list(word2vec, sent)
root = build_recursive(test_tree)
root.draw()

# Plot the learned W matrix
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.set_aspect('equal')
# plt.imshow(W.get_value(), interpolation='nearest', cmap=plt.cm.ocean)
# plt.colorbar()
# plt.show()
