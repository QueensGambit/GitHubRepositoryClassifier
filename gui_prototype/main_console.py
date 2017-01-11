"""
@file: main.py.py
Created on 07.01.2017 18:20
@project: GitHubRepositoryClassifier

@author: Anonym

Sample usage of the repository-classifier
"""

from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent

import sys


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    InputOutputAgent.setWithToken(False)
    repoClassifier = RepositoryClassifier(bUseStringFeatures=False)

    # strFilenameCSV = 'example_repos.csv'
    strFilenameCSV = 'additional_data_sets_cleaned.csv'

    lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData('/data/csv/' + strFilenameCSV)
    repoClassifier.trainModel(lstTrainData, lstTrainLabels)
    repoClassifier.exportModelToFile()
    # repoClassifier.loadModelFromFile()
    repoClassifier.predictResultsAndCompare()

    print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
    repoClassifier.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
    # pobox/overwatch
    # pobox
    repoClassifier.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
    repoClassifier.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')


if __name__ == "__main__":
    main()