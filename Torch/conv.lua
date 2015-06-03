require('torch')

vecs = torch.load('corpus_vecs.th', 'ascii')
vecs_test = torch.load('corpus_vecs_test.th', 'ascii')
labels = torch.load('corpus_labels.th', 'ascii')
labels_test = torch.load('corpus_labels_test.th', 'ascii')

-- Save in numpy format for Python classifier tests
local py = require('fb.python')
py.exec([=[
import numpy as np
f = open('corpus_vecs.npy', 'w+')
np.save(f, vecs)
f = open('corpus_vecs_test.npy', 'w+')
np.save(f, vecs_test)
f = open('corpus_labels.npy', 'w+')
np.save(f, labels)
f = open('corpus_labels_test.npy', 'w+')
np.save(f, labels_test)
]=], {vecs = vecs, vecs_test = vecs_test, labels = labels, labels_test = labels_test})
