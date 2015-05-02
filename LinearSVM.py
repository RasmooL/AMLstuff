from sklearn import svm
from sklearn import lda
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

# Load corpus vectors
data = np.load('Torch/corpus_vecs.npy') # n_samples x n_features
data = np.vstack(data)

# Load labels and encode each class as a number
labels = np.load('Torch/corpus_labels.npy')
labels = np.array( map(lambda (x,y): x, labels) )
labels = LabelEncoder().fit_transform(labels)

# Split into training and test set
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.3)#, random_state=42)

# Fit classifiers
#clda = lda.LDA()
#clda.fit(data_train, labels_train)

csvm = svm.LinearSVC()
csvm.fit(data_train, labels_train)

# Print scores
#print("LDA train classification % = " + str(clda.score(data_train, labels_train) * 100))
#print("LDA test classification % = " + str(clda.score(data_test, labels_test) * 100))
print("SVM train classification % = " + str(csvm.score(data_train, labels_train) * 100))
print("SVM test classification % = " + str(csvm.score(data_test, labels_test) * 100))
print("Best guess % = " + str( float(Counter(labels_train).most_common(1)[0][1]) / len(labels_train) * 100))
