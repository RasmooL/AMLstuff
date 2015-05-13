import json
import numpy as np
from glove import Glove, Corpus

no_comp = 2


# Read yahoo data
uris = []
questions = []
answers = []
cats = []
with open('small_test.txt', 'r') as file:
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
corpus_model.fit(get_lines(), window=4)

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

# Train GloVe model
glove = Glove(no_components = no_comp, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=100, no_threads=4, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

# Save
with open('small_test.glove', 'w+') as file:
    file.write('%i %i \n' % (len(glove.dictionary), no_comp))
    for (word, idx) in glove.dictionary.iteritems():
        file.write('%s %s \n' % (word, ' '.join(str(n) for n in glove.word_vectors[idx])))
