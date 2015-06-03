import os
import json
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import ProjectedGradientNMF
from collections import Counter
import numpy as np


# Read yahoo data
#uris = []
#questions = []
answers_train = []
answers_test = []
cats_train = []
cats_test = []
with open('yahoo_train.txt', 'r') as file:
    for line in file:
        d = json.loads(line)

        #uris.append(d[0])
        #questions.append(d[1])
        answers_train.append(d[2])
        cats_train.append(d[3])

with open('yahoo_test.txt', 'r') as file:
    for line in file:
        d = json.loads(line)

        #uris.append(d[0])
        #questions.append(d[1])
        answers_test.append(d[2])
        cats_test.append(d[3])

# Encode category labels to numbers
#le = LabelEncoder()
#cats_train = le.fit_transform(cats_train)
#cats_test = le.transform(cats_test)

# Split into training and test
#answers_train, answers_test, cats_train, cats_test = train_test_split(answers, cats, test_size = 0.3)#, random_state=42)

# Word counts
count_vect = CountVectorizer(stop_words = 'english')
answers_train = count_vect.fit_transform(answers_train)
answers_test = count_vect.transform(answers_test)

# Tf-idf
tfidf_transformer = TfidfTransformer()
answers_train = tfidf_transformer.fit_transform(answers_train)
answers_test = tfidf_transformer.transform(answers_test)

# NMF fit on training set
print("Fitting NMF on training word count matrix with shape" + str(answers_train.shape))
nmf = ProjectedGradientNMF(n_components = 100, max_iter=200)
answers_train = nmf.fit_transform(answers_train)
answers_test = nmf.transform(answers_test)

# Fit SVM classifier
print("Fitting SVM classifier on matrix with shape" + str(answers_train.shape))
svc = svm.LinearSVC()
svc.fit(answers_train, cats_train)

print("SVM train classification %: " + str(svc.score(answers_train, cats_train) * 100))
print("SVM test classification %: " + str(svc.score(answers_test, cats_test) * 100))
mc_label = Counter(cats_train).most_common(1)[0][0]
print("Best guess % = " + str( float(Counter(cats_test)[mc_label]) / len(cats_test) * 100))

# Metrics
np.set_printoptions(linewidth=200, precision=3)
cats_pred = svc.predict(answers_test)
#c = metrics.confusion_matrix(labels_test, csvm.predict(data_test))
precision, recall, fbeta, support = metrics.precision_recall_fscore_support(cats_test, cats_pred)
print(precision)
print(recall)
print(fbeta)
print(support)

# Do TSNE plot
#import matplotlib
#matplotlib.use('Qt4Agg')
#import pylab as plt
#from tsne import bh_sne
#Y = bh_sne(answers_train, -1, 2, 20)
#fig = plt.figure()
#plt.scatter(Y[:,0], Y[:,1], 20, cats_train)
#plt.show()
#plt.savefig('tsne.png')
