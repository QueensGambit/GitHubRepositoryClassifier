from os import path
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats._discrete_distns import skellam_gen

from githubRepo import GithubRepo
from sklearn.neighbors.nearest_centroid import NearestCentroid
from operator import add

from io_agent import InputOutputAgent
from utility_funcs.preprocessing_operations import *
from utility_funcs.count_vectorizer_operations import *
import logging

import matplotlib.pyplot as plt

#logging.basicConfig(level=logging.DEBUG)

iNumCategories = 7
iNumExamples = 5
lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']

#iNumTrainData = iNumExamples * (iNumCategories - 1)
iNumTrainData = 270 #100 #
print("iNumTrainData: ", iNumTrainData)

# initialize a list with None-values
#lstReadmeURL = [None] * iNumCategories * iNumExamples
lstReadmeURL = [None] * iNumTrainData

directory = path.dirname(__file__)

# get the project-directory
strProjectDir = str(Path().resolve().parent)
print('strProjectDir:', strProjectDir)

print(directory)

# strFilenameCSV = 'example_repos.csv'
strFilenameCSV = 'additional_data_sets_cleaned.csv'
# strFilenameCSV = 'additional_data_sets_skipped_rows.csv'

# trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",",
trainData = pd.read_csv(strProjectDir + '/data/csv/' + strFilenameCSV, header=0, delimiter=",",
                        nrows=iNumTrainData) #, skiprows=100)

lstGithubRepo = []



print('~~~~~~~~~~ EXTRACTING FEATURES ~~~~~~~~~~')
for i in range(iNumTrainData):
    # print(trainData["URL"][i])
    # lststrLabelGroup = trainData["URL"][i].split('/')
    # print(lststrLabelGroup[3] + "\t" + lststrLabelGroup[4])

    # lstGithubRepo.append(GithubRepo(lststrLabelGroup[iIndexUser], lststrLabelGroup[iIndexName]))

    # fill the list with GithubRepo-Objects
   # print('string_operations.extractProjectNameUser: ', string_operations.extractProjectNameUser(trainData["URL"][i]))
    lstGithubRepo.append(GithubRepo.fromURL(trainData["URL"][i]))

# fill the train and the label-data
lstTrainData = []
lstTrainLabels = []

lstMeanValues = [0] * 7
i = 0
for tmpGithubRepo in lstGithubRepo:

    #lstMeanValues += tmpGithubRepo.getFeatures()
    lstMeanValues = list(map(add, lstMeanValues, tmpGithubRepo.getFeatures()))

    # find the according label as an intger for the current repository
    # the label is defined in trainData
    lstTrainLabels.append(lstStrCategories.index(trainData["CATEGORY"][i]))
    i += 1


# replace every 0 with 1, otherwise division by 0 occurs
# http://stackoverflow.com/questions/2582138/finding-and-replacing-elements-in-a-list-python
lstMeanValues[:] = [1 if x==0 else x for x in lstMeanValues]

# Divide each element with the number of training data
lstMeanValues[:] = [x / iNumTrainData for x in lstMeanValues]
print('lstMeanValues: ', lstMeanValues)


print('~~~~~~~~~~ GET THE VOCABULARY ~~~~~~~~~~')
strVocabPath = directory + '/vocab/'
# Create vocab-directory if needed directory
if not os.path.exists(strVocabPath):
    os.makedirs(strVocabPath)
strVocabPath += 'vocabList.dump'
lstVoc = initInputParameters(strVocabPath, lstGithubRepo)

print('lstVoc: ', lstVoc)
print('len(lstVoc): ', len(lstVoc))

lstInputFeatures = []
for tmpGithubRepo in lstGithubRepo:
    # fill the Training-Data
    # ordinary integer-attributes
    # lstTrainData.append(tmpGithubRepo.getNormedFeatures(lstMeanValues))

    # lstInputFeatures = tmpGithubRepo.getNormedFeatures(lstMeanValues)
    # with the word occurrence vector
    # lstInputFeatures = lstInputFeatures + (tmpGithubRepo.getWordSparseMatrix(lstVoc))
    # print(tmpGithubRepo.getNormedFeatures(lstMeanValues))
    # print(tmpGithubRepo.getWordSparseMatrix(lstVoc))

    # np.vstack  concates to numpy-arrays
    lstInputFeatures = tmpGithubRepo.getNormedFeatures(lstMeanValues) + tmpGithubRepo.getWordOccurences(lstVoc)

    lstTrainData.append(lstInputFeatures)

print("lstTrainData:")
print(lstTrainData)

print("lstTrainLabels:")
print(lstTrainLabels)

print('~~~~~~~~~~ TRAIN THE MODEL ~~~~~~~~~~')
# train the nearest neighbour-model
clf = NearestCentroid()
clf.fit(lstTrainData, lstTrainLabels)

print('~~~~~~~~~~ PREDICT RESULTS ~~~~~~~~~~')
# classify the result

# as a sample prediction example use an array of 42, 42, ... as an example feature set
iNumbeTrainingFeatures = len(lstGithubRepo[0].getNormedFeatures(lstMeanValues)) + len(lstVoc)
iLabel = int(clf.predict([[42] * iNumbeTrainingFeatures]))

print('iLabel:', iLabel)
print('Prediction for 42,42:', lstStrCategories[iLabel])

iNumOfPredictions = 7
# read the unlabeled data set from a csv
unlabeledData = pd.read_csv(strProjectDir + '/data/csv/unclassified_repos.csv', header=0, delimiter=",", nrows=iNumOfPredictions)

strStopper1 = "="*80
strStopper2 = "-"*80

for i in range(iNumOfPredictions):
    tmpRepo = GithubRepo.fromURL(unlabeledData["URL"][i])
    print(strStopper1)

    lstInputFeatures = tmpRepo.getNormedFeatures(lstMeanValues) + tmpRepo.getWordOccurences(lstVoc)

    # iLabel = int(clf.predict([tmpRepo.getNormedFeatures(lstMeanValues)]))
    iLabel = int(clf.predict([lstInputFeatures]))
    lstOccurence = tmpRepo.getWordOccurences(lstVoc)
    # print('lstOccurence:', lstOccurence)
    printFeatureOccurences(lstVoc, lstOccurence, 0)
    print('len(lstOccurence):', len(lstOccurence))
    print('Prediction for ' + tmpRepo.getName() + ', ' + tmpRepo.getUser() + ': ', end="")
    print(lstStrCategories[iLabel])
    print(strStopper2)
