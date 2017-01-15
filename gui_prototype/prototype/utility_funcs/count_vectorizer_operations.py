"""
@file: count_vectorizer_operations.py.py
Created on 21.12.2016 00:59
@project: GitHubRepositoryClassifier

@author: Anonym

Your description goes here...
"""


def printFeatureOccurences(lstFeatureNames, lstOccurrence, iMinOccurence=1):
    """
    Prints out every feature with it's number occurence

    :param lstFeatureNames:     list of the given features, these are the column of the sparse-matrix
    :param lstOccurrence:       number of occurence of the individual features (has the same size as lstFeatureName
    :param iMinOccurence:       minimum threshold to print out the feature (if set to 0 all features are print out)
    :return:
    """
    assert (len(lstFeatureNames) == len(lstOccurrence))
    # print('{:.10}'.format(lstFeatureNames[i]), end=":\t")

    strStopper1 = "=" * 80
    strStopper2 = "-" * 80

    print(strStopper2)

    print('detected words from the vocabulary:')
    i = 0
    for iTmpOccurrence in lstOccurrence:
        if iTmpOccurrence > iMinOccurence:
            # for more beautiful print layout {:15s} and {:3d} is used
            print('{:15s} {:3f}'.format(lstFeatureNames[i], iTmpOccurrence)) #{:3d} for integers
        i += 1

    print(strStopper2)
