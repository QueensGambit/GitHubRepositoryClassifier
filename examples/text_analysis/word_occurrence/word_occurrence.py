""" infos about CountVectorizer at:
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform
"""

from sklearn.feature_extraction.text import CountVectorizer
from os import path

import numpy as np

d = path.dirname(__file__)
strPath = d + "/lorem_ipsum.txt"

text_file = open(strPath, "r")
# load the single lines to an array
lstLoremIpsumLines = text_file.readlines()

print(lstLoremIpsumLines)

# create a counter which counts the occurrence of each word which is defined in the vocabulary
# by default the vocabulary consists of all words
vectorizer = CountVectorizer(min_df=1)


# return a sparse matrix
# each column is mapped to a specific feature (see lstFeatureNames)
# the value describes the occurrence of the word in the current line
matSparse = vectorizer.fit_transform(lstLoremIpsumLines)

lstFeatureNames = vectorizer.get_feature_names()
lstVoc = vectorizer.vocabulary


print('~~~~~~~~~~ Sparse-Matrix ~~~~~~~~~~')
print(matSparse.toarray())

print('~~~~~~~~~~ lstFeatureNames ~~~~~~~~~~')
print(lstFeatureNames)


print('~~~~~~~~~~ Number-of-total-occurrences ~~~~~~~~~~')

matOccurrence = np.asarray(np.sum(matSparse, axis=0))
# flatten makes a matrix 1 dimensional
lstOccurrence = matOccurrence.flatten()

i = 0
for iTmpOccurrence in lstOccurrence:
    #print('{:.10}'.format(lstFeatureNames[i]), end=":\t")
    #print(iTmpOccurrence)

    # for more beautiful print layout {:15s} and {:3d} is used
    print('{:15s} {:3d}'.format(lstFeatureNames[i], iTmpOccurrence))
    i += 1
