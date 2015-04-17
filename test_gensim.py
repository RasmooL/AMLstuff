from gensim import corpora, models, similarities
from util import *

# Load word embeddings
#word2vec = load_embeddings('vectors.6B.50d.txt')

# Load sentences
sentences = []
with open('corpus','r') as infile:
    count = 0
    for line in infile:
        line = line.decode('ascii', 'ignore')
        line = line.replace('.','')
        line = line.replace('\r','')
        line = line.replace('\n','')
        words = line.split(' ')
        sentence = models.doc2vec.LabeledSentence(words=words, labels='SENT_'+str(count))
        sentences.append(sentence)
        count = count + 1

print sentences[0]
model = models.Doc2Vec.load_word2vec_format('vectors_gensim', binary=False)
print model.most_similar('sir')
