import os
import os.path
import lxml.etree
import string
import progressbar
import json
from itertools import islice
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

datafile = "yahoo_data.txt"
ofile = open(datafile, 'w+')
filename = "manner.xml"
tree = lxml.etree.parse(open(filename, 'r'))

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

docs = tree.xpath('//document')
num_docs = 3000#len(docs)
pbar = progressbar.ProgressBar(maxval=num_docs).start()
for n, element in islice(enumerate(docs), num_docs):
    # Get main category or 'Misc'
    if element.xpath('maincat'):
        maincat = element.xpath('maincat')[0].text
    else:
        maincat = 'Misc'

    # Get sub category or 'Misc'
    if element.xpath('subcat'):
        subcat = element.xpath('subcat')[0].text
    else:
        subcat = 'Misc'

    # Get question and preprocess
    question = preprocess(element.xpath('subject')[0].text)

    # Get answer and preproecss
    bestanswer = preprocess(element.xpath('bestanswer')[0].text)

    # Get uri
    uri = element.xpath('uri')[0].text

    # Write lines
    for l in bestanswer.split('\n'):
        if(len(l.split()) > 2): # Remove single symbols, words or empty lines
            ofile.write(json.dumps([int(uri), question, l, maincat, subcat]) + "\n")


    pbar.update(n)

pbar.finish()
