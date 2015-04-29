import os
import os.path
import lxml.etree
import string
import progressbar
from itertools import islice
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

datadir = "data/"
if not os.path.exists(datadir):
    os.makedirs(datadir)

filename = "manner.xml"
tree = lxml.etree.parse(open(filename))

def preprocess(p):
    p = p.replace('<br />', '') # Stupid HTML line breaks
    p = p.replace('\n', ' ')    # Newlines (= 1 line per file)
    p = p.replace('.', ' . ')   # Space around dot
    p = p.replace(',', ' , ')   # Space around comma
    p = filter(lambda x: x in string.printable, p) # Only printable characters

    punct = '"#$%()*+:;<=>@[\]^_`{|}~'
    p = filter(lambda x: x not in punct, p) # Remove various punctuation
    return p

docs = tree.xpath('//document')
num_docs = len(docs)
pbar = progressbar.ProgressBar(maxval=num_docs).start()
for n, element in islice(enumerate(docs), num_docs):
    # Get category or 'Misc'
    if element.xpath('maincat'):
        maincat = element.xpath('maincat')[0].text
    else:
        maincat = 'Misc'
    bestanswer = element.xpath('bestanswer')[0].text
    uri = element.xpath('uri')[0].text

    # Create folder
    if not os.path.exists(datadir + maincat):
        os.makedirs(datadir + maincat)

    # Preprocess best answer paragraph
    bestanswer = preprocess(bestanswer)

    # Create file and write best answer
    with open(datadir + maincat + "/" + uri + ".txt", 'w+') as f:
        f.write(bestanswer)

    pbar.update(n)

pbar.finish()
