import gensim
import os
import json

datadir = 'data'
size = 100
min_count = 1
window = 5
workers = 4
negative = 8
epochs = 2

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

sentences = get_lines()
#model = gensim.models.Word2Vec(sentences=None, size=size, min_count=min_count, window=window, workers=workers, negative=negative, iter=epochs)
model = gensim.models.Word2Vec.load_word2vec_format('vectors.6B.100d.txt', binary=False)

print("Number of tokens in vocabulary: " + str(len(model.index2word)))

model.build_vocab(sentences)

print("Number of tokens in vocabulary: " + str(len(model.index2word)))

model.train(sentences)

#print(model.most_similar('computer'))

model.save_word2vec_format('model.word2vec')
