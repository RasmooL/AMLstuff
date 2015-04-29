from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = word2vec.Word2Vec.load_word2vec_format('model.word2vec')

#more_examples = ["he his she", "big bigger bad", "going went being"]
#for example in more_examples:
#     a, b, x = example.split()
#     predicted = model.most_similar([x, b], [a])[0][0]
#     print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
#print(model.doesnt_match("breakfast cereal dinner lunch".split()))
print(model.most_similar("."))
