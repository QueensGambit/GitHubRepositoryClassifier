# interfaces / abstract classes in python
from abc import ABCMeta, abstractmethod

class Interface_RepoClassifier(metaclass=ABCMeta):

    @abstractmethod
    def loadTrainingData(self, strProjPathFileNameCSV):
        """
        abstract method
        The classifier loads the sample data from a given csv-file

        :param strProjPathFileNameCSV: path to the csv-file
        :return:
        """
        pass

    @abstractmethod
    def trainModel(self, lstTrainData, lstTrainLabels):
        """
        abstract method
        The model shall be trained via supervised learning

        :param lstTrainData: matrix of the training data
        :param lstTrainLabels: list of the associated labels
        :return:
        """
        pass

    @abstractmethod
    def plotTheResult(self):
        """
        abstract method
        A plot in which the classification is illustrated

        :return:
        """
        pass

    @abstractmethod
    def exportModelToFile(self):
        """
        abstract method
        Export the model and all prequisites to the directory model/

        :return:
        """
        pass

    @abstractmethod
    def loadModelFromFile(self):
        """
        abstract method
        Loading of the exported model

        :return:
        """
        pass

    @abstractmethod
    def predictResultsAndCompare(self, strProjPathFileNameCSV):
        """
        abstract method
        Predict a given csv-file and compare the result with the manual classification

        :param strProjPathFileNameCSV: path to the csv-file
        :return:
        """
        pass

    @abstractmethod
    def predictCategoryFromOwnerRepoName(self, strUser, strRepoName):
        """
        abstract method
        Predict the category for a repository

        :param strUser:
        :param strRepoName:
        :return:
        """
        pass

    @abstractmethod
    def predictCategoryFromURL(self, strGitHubRepoURL):
        """
        abstract method
        Predict the category

        :param strGitHubRepoURL:
        :return:
        """
        pass
