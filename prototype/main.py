"""
@file: main.py.py
Created on 07.01.2017 18:20
@project: GitHubRepositoryClassifier

@author: Anonym

Sample usage of the repository-classifier
"""

from prototype import RepoClassifierNearestNeighbour


repoClassNN = RepoClassifierNearestNeighbour()


# strFilenameCSV = 'example_repos.csv'
strFilenameCSV = 'additional_data_sets_cleaned.csv'

repoClassNN.trainModel('/data/csv/' + strFilenameCSV)
repoClassNN.exportModelToFile()
# repoClassNN.loadModelFromFile()
repoClassNN.predictResultsAndCompare()

print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
repoClassNN.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
# pobox/overwatch
# pobox
repoClassNN.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
repoClassNN.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')