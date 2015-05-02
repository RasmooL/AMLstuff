import os
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count = 0
data = []
questions = []
for root, subdirs, files in os.walk('data'):
    for fname in files:
        if fname.endswith(".txt"):
            lines = open(os.path.join(root, fname)).readlines()
            data.append('\n'.join(lines))
            count = count + 1
        elif fname.endswith(".q"):
            lines = open(os.path.join(root, fname)).readlines()
            questions.append('\n'.join(lines))

# Word counts
count_vect = CountVectorizer(stop_words = 'english')
train_counts = count_vect.fit_transform(data)

# Tf-idf
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

sumrrank = 0
num = count
for n in xrange(num):
    scores = {} # Dictionary
    # rank documents for q
    q = questions[n]
    q_col = count_vect.transform([q]).indices # Non-zero indices in query

    # Calculate score for each answer
    for i in xrange(count):
        d = train_tfidf[i].toarray()[0] # tfidf vector for answer i
        scores[i] = 0

        # Sum scores for answer i
        for col in q_col:
            scores[i] = scores[i] + d[col] # Add answer tfidf value
    ranks = sorted(scores.items(), key = lambda x: x[1], reverse = True) # Sort corpus

    # Find reciprocal rank of correct answer
    rank = -1
    rrank = 0
    for i in xrange(count):
        if n == ranks[i][0]:
            rank = i + 1
            break
    if rank != -1:
        rrank = 1.0 / rank

    # Sum up reciprocal ranks
    sumrrank = sumrrank + rrank

print("MRR = " + str(sumrrank/num))
