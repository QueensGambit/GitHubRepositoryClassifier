from prototype.github_repo import GithubRepo
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from prototype.utility_funcs.io_agent import InputOutputAgent
from os import path
from prototype.definitions import githubLanguages

# read all input files
print('language list:')
print(githubLanguages.lstLanguages)
print('.................')

emojiRepo = GithubRepo("WebpageFX", "emoji-cheat-sheet.com")


ourRepo = GithubRepo("QueensGambit", "Barcode-App")

print('->', emojiRepo.intFeatures.iSubscriberCount)

print(emojiRepo)


# train
lstTrainData = np.array([emojiRepo.getIntegerFeatures(), ourRepo.getIntegerFeatures()])
# lstTrainData = np.array([[42, 42], [7, 543]])
lstTrainLabels = np.array([0, 1])
print('trainData:', lstTrainData)
print('trainLabels:', lstTrainLabels)

clf = NearestCentroid()
clf.fit(lstTrainData, lstTrainLabels)

strUser = 'QueensGambit'
strName = 'Barcode-App'
d = path.dirname(__file__)
strPathJSON = d + '/json/' + strUser + '_' + strName + '.json'
print('strPathJSON:', strPathJSON)


print(clf.predict([[42] * len(ourRepo.getIntegerFeatures())]))

print('Prediciton for Barocde-App:', clf.predict([ourRepo.getIntegerFeatures()]))

# Lucifier129_Lucifier129.github.io
io = InputOutputAgent('Lucifier129', 'Lucifier129.github.io')
tmpRepo = GithubRepo('Lucifier129', 'Lucifier129.github.io')
strReadme = tmpRepo.getReadme()
# strReadme = io.getReadme()
print('strReadme: ', strReadme)