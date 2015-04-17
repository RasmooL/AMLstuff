import os, os.path
import string

dir = 'train/'
thingstofilter = '!"#$%()*+,-:;<=>?@[\]^_`{|}~'

files = os.listdir(dir)
for fname in files:
    if not '.snt' in fname:
        continue
    fromf = open(dir + fname, 'r')

    title = fromf.readline()
    title = title.replace('\r\n', '')
    title = filter(lambda x: x not in string.punctuation, title)
    print(title)
    tof = open(dir + title + '.txt', 'w')

    for line in fromf:
        line = line.replace('.\r\n', '')
        line = line.replace('.\n', '')
        #for p in punctuation:
        #    line = line.replace(p, '')
        line = filter(lambda x: x in string.printable, line)
        line = filter(lambda x: x not in thingstofilter, line)
        tof.write(line + '\n')
