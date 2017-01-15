"""
@file: main.py.py
Created on 07.01.2017 18:20
@project: GitHubRepositoryClassifier

@author: Anonym

Sample usage of the repository-classifier
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.semi_supervised import label_propagation
from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent
from prototype.github_repo import GithubRepo
from matplotlib.colors import colorConverter

from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
import matplotlib.patches as mpatches

import sys
from prototype.definitions.categories import CategoryStr


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    InputOutputAgent.setWithToken(True)
    repoClassifier = RepositoryClassifier(bUseStringFeatures=True)

    # strFilenameCSV = 'example_repos.csv'
    strFilenameCSV = 'additional_data_sets_cleaned.csv'

    lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData('/data/csv/' + strFilenameCSV)
    repoClassifier.trainModel(lstTrainData, lstTrainLabels)
    repoClassifier.exportModelToFile()
    # clf, lstMeanValues, matIntegerTrainingData, lstTrainLabels, lstTrainData, normalizer, normalizerIntegerAttr = repoClassifier.loadModelFromFile()
    repoClassifier.predictResultsAndCompare()

    print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
    iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
    # pobox/overwatch
    # pobox
    repoClassifier.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
    repoClassifier.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')
    print('lstNormedInputFeatures')




#### RESULTS
# fPredictionRes: 0.612903225806
# fAccuracy:  61.2903225806 %
# NearestCentroid()
# fPredictionRes: 0.58064516129
# fAccuracy:  58.064516129 %
#
#
# KNeighborsClassifier()
# fPredictionRes: 0.41935483871
# fAccuracy:  41.935483871 %
#
#
# RadiusNeighborsClassifier()
#
#
# -- > without removing stop words and with length > 3
# fPredictionRes: 0.645161290323
# fAccuracy:  64.5161290323 %
# --> this is the best result but it doesn't feel right,
#  due to it's randomness of stopper words
#
# ---> lenght < 2 and with removing stop words
# fPredictionRes: 0.612903225806
# fAccuracy:  61.2903225806 %


if __name__ == "__main__":
    main()