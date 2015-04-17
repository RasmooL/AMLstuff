import sys
import numpy as np
import theano
import theano.tensor as T
from util import *

# Load word embeddings
word2vec = load_embeddings('vectors.6B.50d.txt')

cfile = open('corpus', 'r')
for line in cfile:
    line = line.split(' ')
    line_embed = string_embedding(word2vec, line)
    
