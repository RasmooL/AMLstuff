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

tree = T.vector('tree')
# Implementation for theano.scan
def build_scan(tree):
    if not tree:
        raise Exception("Empty tree!")

    #out, upd = theano.scan( fn=blah,
    #                        n_steps = len(tree) - 1)
    return tree, theano.scan_module.until(len(tree) == 1) # Recurse until root node is found

outputs, updates = theano.scan( fn=build_scan,
                                sequences=None,
                                outputs_info=None,
                                non_sequences=tree,
                                n_steps=1000 # Maximum number of recursions (because of until)
)

sent = ['the', 'cat', 'sat', 'on', 'the', 'mat']
test_tree = make_node_list(word2vec, sent)
root = build_recursive(test_tree)
root.draw()
