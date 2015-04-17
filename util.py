import numpy as np
import collections
from node import Node

# Loads word embeddings into dictionary
def load_embeddings(fname):
    infile = open(fname, 'r')
    word2vec = dict()

    for line in infile:
    	line = line.split(' ')
    	word = line.pop(0)
    	vec = [float(x) for x in line]
    	word2vec[word] = vec

    return word2vec

# Get string embedding
def string_embedding(dict, string):
    try:
        embed = map(lambda word: dict[word], string)
    except KeyError as e:
        print "Error: Word '" + e.args[0] + "' is not in dictionary."
        exit()
    return embed

# Numpy implementation of cosine similarity
def cosine_similarity(vec1, vec2):
	return np.dot(vec1, vec2) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))

# List of words -> list of Nodes with embedding values
def make_node_list(dict, words):
    embed = string_embedding(dict, words)
    l = []
    for w,e in zip(words,embed):
        l.append( Node(name=w, value=e) )

    return l

# Find the n closest [word, embed] pairs to vec in dict using distance function dist
def n_closest_vec(dict, vec, n=1, dist=lambda x,y: np.linalg.norm(np.array(x) - np.array(y))):
    best_dist = np.inf
    best = collections.deque(n*[0], n)
    for word,embed in dict.iteritems():
        d = dist(vec, embed)
        if d < best_dist:
            best_dist = d
            best.appendleft([word, embed])
    return sorted(best, key=lambda x: dist(vec,x[1]))
