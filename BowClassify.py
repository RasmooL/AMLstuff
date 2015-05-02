import os
import json
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import ProjectedGradientNMF
from collections import Counter
import numpy as np


# Read yahoo data
uris = []
questions = []
answers = []
cats = []
with open('yahoo_data.txt', 'r') as file:
    for line in file:
        d = json.loads(line)

        uris.append(d[0])
        questions.append(d[1])
        answers.append(d[2])
        cats.append(d[3])

# Encode category labels to numbers
cats = LabelEncoder().fit_transform(cats)

# Word counts
count_vect = CountVectorizer(stop_words = 'english')
counts = count_vect.fit_transform(answers)

# Tf-idf
#tfidf_transformer = TfidfTransformer()
#train_tfidf = tfidf_transformer.fit_transform(train_counts)

# Split into training and test
answers_train, answers_test, cats_train, cats_test = train_test_split(counts, cats, test_size = 0.3)#, random_state=42)

# NMF fit on training set
print("Fitting NMF on training word count matrix with shape" + str(answers_train.shape))
#nmf = ProjectedGradientNMF(n_components = 50)
#answers_train = nmf.fit_transform(answers_train)
answers_train = np.load('answers_train.npy')

# NMF transform test set
#answers_test = nmf.transform(answers_test)

# Fit SVM classifier
print("Fitting SVM classifier on matrix with shape" + str(answers_train.shape))
svc = svm.LinearSVC()
#svc.fit(answers_train, cats_train)

#print("SVM train classification %: " + str(svc.score(answers_train, cats_train) * 100))
#print("SVM test classification %: " + str(svc.score(answers_test, cats_test) * 100))
#print("Best guess %: " + str( float(Counter(cats_train).most_common(1)[0][1]) / len(cats_train) * 100))

# Do TSNE plot
import matplotlib
matplotlib.use('Qt4Agg')
import pylab as plt
from tsne import bh_sne
Y = bh_sne(answers_train, -1, 2, 20)
fig = plt.figure()
plt.scatter(Y[:,0], Y[:,1], 20, cats_train)
plt.show()
#plt.savefig('tsne.png')
