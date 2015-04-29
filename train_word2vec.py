import gensim
import os

datadir = 'data'
size = 50
min_count = 4
window = 10
workers = 8
negative = 1

class SentenceIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, subdirs, files in os.walk(self.dirname):
            for fname in files:
                for line in open(os.path.join(root, fname)):
                    yield line.split()

sentences = SentenceIterator(datadir)
model = gensim.models.Word2Vec(sentences=None, size=size, min_count=min_count, window=window, workers=workers, negative=negative)

model.build_vocab(sentences)

print("Number of tokens in vocabulary: " + str(len(model.index2word)))

model.train(sentences)

#print(model.most_similar('computer'))

model.save_word2vec_format('model.word2vec')
