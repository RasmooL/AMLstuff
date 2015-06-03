from sklearn import svm
from sklearn import lda
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

# Load corpus vectors
data_train = np.load('Torch/corpus_vecs.npy') # n_samples x n_features
data_train = np.vstack(data_train)
data_test = np.load('Torch/corpus_vecs_test.npy')
data_test = np.vstack(data_test)

# Load labels and encode each class as a number
labels_train = np.load('Torch/corpus_labels.npy')
labels_test = np.load('Torch/corpus_labels_test.npy')
#   labels = np.array( map(lambda (x,y): x, labels) )
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.transform(labels_test)

# Truncate training data
#data_train = data_train[0:2000]
#labels_train = labels_train[0:2000]

# Split into training and test set
#data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.3)#, random_state=42)

# Fit classifiers
#clda = lda.LDA()
#clda.fit(data_train, labels_train)

#cnb = GaussianNB()
#cnb.fit(data_train, labels_train)

#csvm = svm.SVC(kernel='poly', degree=3, class_weight='auto')
csvm = svm.LinearSVC()
csvm.fit(data_train, labels_train)

# Print scores
#print("LDA train classification % = " + str(clda.score(data_train, labels_train) * 100))
#print("LDA test classification % = " + str(clda.score(data_test, labels_test) * 100))
#print("NB train classification % = " + str(cnb.score(data_train, labels_train) * 100))
#print("NB test classification % = " + str(cnb.score(data_test, labels_test) * 100))
print("SVM train classification % = " + str(csvm.score(data_train, labels_train) * 100))
print("SVM test classification % = " + str(csvm.score(data_test, labels_test) * 100))
mc_label = Counter(labels_train).most_common(1)[0][0]
print("Best guess % = " + str( float(Counter(labels_test)[mc_label]) / len(labels_test) * 100))

# Metrics
np.set_printoptions(linewidth=200, precision=3)
labels_pred = csvm.predict(data_test)
#c = metrics.confusion_matrix(labels_test, csvm.predict(data_test))
precision, recall, fbeta, support = metrics.precision_recall_fscore_support(labels_test, labels_pred)
print(le.classes_)
print(precision)
print(recall)
print(fbeta)
print(support)
