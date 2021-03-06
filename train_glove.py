import json
import numpy as np
from glove import Glove, Corpus
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

no_comp = 100

# Read yahoo data
uris = []
questions = []
answers = []
cats = []
with open('yahoo_train.txt', 'r') as file:
    for line in file:
        d = json.loads(line)

        uris.append(d[0])
        questions.append(d[1])
        answers.append(d[2])
        cats.append(d[3])

def get_lines():
    for a in answers:
        yield a.split()

# Build the corpus dictionary and cooccurence matrix
corpus_model = Corpus()
corpus_model.fit(get_lines(), window=8)

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

# Train GloVe model
#glove = Glove(no_components = no_comp, learning_rate=0.05)
glove = Glove.load_stanford('vectors.6B.100d.txt')
glove.fit(corpus_model.matrix, epochs=10, no_threads=4, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

# Save
with open('model.glove', 'w+') as file:
    file.write('%i %i \n' % (len(glove.dictionary), no_comp))
    for (word, idx) in glove.dictionary.iteritems():
        file.write('%s %s \n' % (word, ' '.join(str(n) for n in glove.word_vectors[idx])))
