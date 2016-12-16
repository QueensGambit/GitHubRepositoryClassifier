from os import path
import pandas as pd
import numpy as np
from githubRepo import GithubRepo
from sklearn.neighbors.nearest_centroid import NearestCentroid
from operator import add

iNumCategories = 7
iNumExamples = 5
lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHERS']

# initialize a list with None-values
lstReadmeURL = [None] * iNumCategories * iNumExamples

directory = path.dirname(__file__)
print(directory)

trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",",
                        nrows=iNumExamples * (iNumCategories - 1))

iNumTrainData = iNumExamples * (iNumCategories - 1)
print("iNumTrainData: ", iNumTrainData)

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


for tmpGithubRepo in lstGithubRepo:
    # train the nearest neighbour-model
    lstTrainData.append(tmpGithubRepo.getNormedFeatures(lstMeanValues))



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

# as a sample prediction example use an array of 42, 42, ... as an example feature set
iLabel = int(clf.predict([[42]*len(lstGithubRepo[0].getNormedFeatures(lstMeanValues))]))
print('iLabel:', iLabel)
print('Prediction for 42,42:', lstStrCategories[iLabel])


unlabeledData = pd.read_csv(directory + "/unclassified_repos.csv", header=0, delimiter=",", nrows=4)

for i in range(4):
    tmpRepo = GithubRepo.fromURL(unlabeledData["URL"][i])
    iLabel = int(clf.predict([tmpRepo.getNormedFeatures(lstMeanValues)]))

    print('Prediction for ' + tmpRepo.getName() + ', ' + tmpRepo.getUser() + ': ', end="")
    print(lstStrCategories[iLabel])
