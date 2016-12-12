from os import path
import pandas as pd
import numpy as np
from githubRepo import GithubRepo
from sklearn.neighbors.nearest_centroid import NearestCentroid


iNumCategories = 7
iNumExamples = 5
lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHERS']

# initialize a list with None-values
lstReadmeURL = [None] * iNumCategories * iNumExamples

directory = path.dirname(__file__)
print(directory)

trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",",
                        nrows=iNumExamples * (iNumCategories - 1))

print(iNumExamples * (iNumCategories - 1))

lstGithubRepo = []


print('~~~~~~~~~~ EXTRACTING FEATURES ~~~~~~~~~~')
for i in range(iNumExamples * (iNumCategories - 1)):
    # print(trainData["URL"][i])
    lststrLabelGroup = trainData["URL"][i].split('/')
    # print(lststrLabelGroup[3] + "\t" + lststrLabelGroup[4])

    # fill the list with GithubRepo-Objects
    lstGithubRepo.append(GithubRepo(lststrLabelGroup[3], lststrLabelGroup[4]))

# fill the train and the label-data
lstTrainData = []
lstTrainLabels = []

i = 0
for tmpGithubRepo in lstGithubRepo:

    # train the nearest neighbour-model
    lstTrainData.append(tmpGithubRepo.getFeatures())

    # find the according label as an intger for the current repository
    # the label is defined in trainData
    lstTrainLabels.append(lstStrCategories.index(trainData["CATEGORY"][i]))
    i += 1

print("lstTrainData:")
print(lstTrainData)

print("lstTrainLabels:")
print(lstTrainLabels)

print('~~~~~~~~~~ TRAIN THE MODEL ~~~~~~~~~~')
# train the model
clf = NearestCentroid()
clf.fit(lstTrainData, lstTrainLabels)


print('~~~~~~~~~~ PREDICT RESULTS ~~~~~~~~~~')
# classify the result
iLabel = int(clf.predict([[42]*len(lstGithubRepo[0].getFeatures())]))
print('iLabel:', iLabel)
print('Prediction for 42,42:', lstStrCategories[iLabel])

repoBarcodeApp = GithubRepo('QueensGambit', 'Barcode-App')
iLabel = int(clf.predict([repoBarcodeApp.getFeatures()]))
print('Prediciton for Barocde-App:', iLabel, lstStrCategories[iLabel])

repoAtom = GithubRepo('atom', 'atom')
iLabel = int(clf.predict([repoAtom.getFeatures()]))
print('Prediciton for repoAtom-App:', iLabel, lstStrCategories[iLabel])

repoAtom = GithubRepo('mongodb', 'docs')
iLabel = int(clf.predict([repoAtom.getFeatures()]))
print('Prediciton for mongodb-Docs:', iLabel, lstStrCategories[iLabel])
