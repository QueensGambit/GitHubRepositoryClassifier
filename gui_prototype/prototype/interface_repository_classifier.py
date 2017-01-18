# interfaces / abstract classes in python
from abc import ABCMeta, abstractmethod

class InterfaceRepoClassifier(metaclass=ABCMeta):

    @abstractmethod
    def loadTrainingData(self, strProjPathFileNameCSV):
        pass

    @abstractmethod
    def trainModel(self, lstTrainData, lstTrainLabels):
        pass

    @abstractmethod
    def plotTheResult(self): #,lstTrainData, lstTrainLabels):
        pass

    @abstractmethod
    def exportModelToFile(self):
        pass

    @abstractmethod
    def loadModelFromFile(self):
        pass

    @abstractmethod
    def predictResultsAndCompare(self, strProjPathFileNameCSV):
        pass

    @abstractmethod
    def predictCategoryFromOwnerRepoName(self, strUser, strRepoName):
        pass

    @abstractmethod
    def predictCategoryFromURL(self, strGitHubRepoURL):
        pass
