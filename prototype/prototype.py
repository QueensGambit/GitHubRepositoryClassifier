from os import path
import pandas as pd


iNumCategories = 7
iNumExamples = 5
lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHERS']

# initialize a list with None-values
lstReadmeURL = [None]*iNumCategories*iNumExamples

directory = path.dirname(__file__)
print(directory)

trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",", nrows=iNumExamples*(iNumCategories-1))

print(iNumExamples*(iNumCategories-1))

for i in range(iNumExamples*(iNumCategories-1)):
    print(trainData["URL"][i])
    lststrLabelGroup = trainData["URL"][i].split('/')
    print(lststrLabelGroup[3] + "\t" + lststrLabelGroup[4])

