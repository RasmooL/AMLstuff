import os
import os.path
import lxml.etree
import string
import progressbar
import json
from itertools import islice
from sklearn.preprocessing import LabelEncoder
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

trainname = "yahoo_train.txt"
testname = "yahoo_test.txt"
trainfile = open(trainname, 'w+')
testfile = open(testname, 'w+')
filename = "manner.xml"
tree = lxml.etree.parse(open(filename, 'r'))

docs = tree.xpath('//document')
num_train = 8000
num_test = 2000
num = num_train + num_test
cats = ['Home & Garden', 'Travel', 'Science & Mathematics', 'Pregnancy & Parenting', 'Food & Drink']
cats2num = True
le = LabelEncoder()
if cats2num: # Numeric categories/labels
    le.fit(cats)


def preprocess(p):
    p = p.replace('<br />', '\n') # Stupid HTML line breaks
    p = p.replace('.', ' . \n')   # Space around dot
    p = p.replace(',', ' , ')     # Space around comma
    p = p.replace('?', ' ? ')     # Space around question mark
    p = p.replace('!', ' ! ')     # Space around exclamation mark
    p = filter(lambda x: x in string.printable, p) # Only printable characters

    punct = '"#$%()*+:;<=>@[\]/^_`{|}~'
    p = filter(lambda x: x not in punct, p) # Remove various punctuation
    return p.lower() # Lower-case

pbar = progressbar.ProgressBar(maxval = len(docs)).start()
count_train = 0
count_test = 0
for n, element in enumerate(docs):
    # Terminate when enough test
    if count_test > num_test:
        break

    # Get main category or 'Misc'
    if element.xpath('maincat'):
        maincat = element.xpath('maincat')[0].text
    else:
        maincat = 'Misc'

    # Only use documents from categories 'cats'
    if not maincat in cats:
        continue

    # Numeric label?
    if cats2num:
        maincat = le.transform(maincat) + 1

    # Get sub category or 'Misc'
    if element.xpath('subcat'):
        subcat = element.xpath('subcat')[0].text
    else:
        subcat = 'Misc'


    # Get question and preprocess
    question = preprocess(element.xpath('subject')[0].text)

    # Get answer and preprocess
    bestanswer = preprocess(element.xpath('bestanswer')[0].text)

    # Get uri
    uri = element.xpath('uri')[0].text

    # Write lines
    for l in bestanswer.split('\n'):
        if(len(l.split()) > 2): # Remove single symbols, words or empty lines
            if count_train < num_train:
                trainfile.write(json.dumps([int(uri), question, l, maincat, subcat]) + "\n")
                count_train = count_train + 1
            elif count_test < num_test:
                testfile.write(json.dumps([int(uri), question, l, maincat, subcat]) + "\n")
                count_test = count_test + 1

    pbar.update(n)

pbar.finish()
print(count_train)
print(count_test)
