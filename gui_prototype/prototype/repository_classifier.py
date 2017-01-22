from operator import add
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors.nearest_centroid import NearestCentroid

from os import path

from .utility_funcs.preprocessing_operations import initInputParameters, readVocabFromFile
from .interface_repository_classifier import Interface_RepoClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import  LogisticRegression

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
#logging.basicConfig(level=logging.DEBUG)
from .utility_funcs.io_agent import InputOutputAgent
# import prototype.github_repo
from .github_repo import GithubRepo
from numpy import array
from pathlib import Path
import math


class RepositoryClassifier(Interface_RepoClassifier):

    def __init__(self, bUseStringFeatures=True):
        """
        constructor which initializes member variables

        """

        self.bModelLoaded = False
        self.bModelTrained = False
        self.clf = None
        self.lstMeanValues = None
        self.lstVoc = None
        self.stdScaler = None
        self.lstTrainLabels = None
        self.lstTrainData = None
        self.normalizer = None
        self.bUseCentroids = True
        self.normalizerIntegerAttr = None
        # self.scaler = None
        self.lstTrainDataRaw = None

        self.lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']

        self.directory = path.dirname(__file__)
        print(self.directory)

        self.bUseStringFeatures = bUseStringFeatures

        # get the project-directory
        self.strProjectDir = str(Path().resolve().parent)

        print('strProjectDir:', self.strProjectDir)

        self.strModelPath = self.directory + '/model/'

        # Create model-directory if needed
        if not os.path.exists(self.strModelPath):
            os.makedirs(self.strModelPath)
        self.strModelFileName = 'RepositoryClassifier.pkl'
        self.strLstMeanValuesFileName = 'lstMeanValues.pkl'
        self.strMatIntegerTrainingData = 'matIntegerTrainingData.pkl'
        self.strLstTrainLabels = 'lstTrainLabels.pkl'
        self.strLstTrainData = 'lstTrainData.pkl'
        self.strNormalizer = 'normalizer.pkl'
        self.strNormalizerIntegerAttr = 'normalizerIntegerAttr.pkl'
        self.strLstTrainDataRaw = 'lstTrainDataRaw.pkl'

        self.iNumCategories = len(self.lstStrCategories)

        self.matIntegerTrainingData = []

    def loadTrainingData(self, strProjPathFileNameCSV ='/data/csv/additional_data_sets_cleaned.csv', externalpath=None):
        """
        trains the model with a given csv-file. the csv file must have 2 columns URL and CATEGORY.
        the URL is given in the form 'https://github.com/owner/repository-name'
        the CATEGORY is given by one of these options 'DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER'

        :param strProjPathFileNameCSV: file path relative to the project-path where the csv-file is stored
        :return: self.lstTrainData (the scaled and normed data with which the model was trained with),
         self.lstTrainLabels (the used training labels)
        """
        trainData = None

        if externalpath is None:
            trainData = pd.read_csv(self.directory + strProjPathFileNameCSV, header=0, delimiter=",")
        else:
            trainData = pd.read_csv(strProjPathFileNameCSV, header=0, delimiter=",")

        iNumTrainData = len(trainData.index)
        print("iNumTrainData: ", iNumTrainData)

        print('~~~~~~~~~~ EXTRACTING FEATURES ~~~~~~~~~~')

        lstGithubRepo = []

        for i in range(iNumTrainData):
            # fill the list with GithubRepo-Objects
            lstGithubRepo.append(GithubRepo.fromURL(trainData["URL"][i]))

        # fill the train and the label-data
        self.lstTrainData = []
        self.lstTrainDataRaw = []
        self.lstTrainLabels = []

        print('~~~~~~~~~~ CALCULATE THE MEAN VALUES ~~~~~~~~~~')
        self.lstMeanValues = [0] * 7
        i = 0
        for tmpRepo in lstGithubRepo:

            # lstMeanValues += tmpGithubRepo.getIntegerFeatures()
            self.lstMeanValues = list(map(add, self.lstMeanValues, tmpRepo.getIntegerFeatures()))

            # find the according label as an intger for the current repository
            # the label is defined in trainData
            self.lstTrainLabels.append(self.lstStrCategories.index(trainData["CATEGORY"][i]))
            i += 1

        # Divide each element with the number of training data
        self.lstMeanValues[:] = [x / iNumTrainData for x in self.lstMeanValues]

        print('lstMeanValues: ', self.lstMeanValues)

        print('~~~~~~~~~~ GET THE VOCABULARY ~~~~~~~~~~')
        strVocabPath = self.directory + '/vocab/'
        # Create vocab-directory if needed directory
        if not os.path.exists(strVocabPath):
            os.makedirs(strVocabPath)
        strVocabPath += 'vocabList.dump'
        self.lstVoc = initInputParameters(strVocabPath, lstGithubRepo)

        print('lstVoc: ', self.lstVoc)
        print('len(lstVoc): ', len(self.lstVoc))

        lstInputFeatures = []
        lstInputFeaturesRaw = []

        for tmpRepo in lstGithubRepo:

            lstIntegerAttributes = tmpRepo.getNormedFeatures(self.lstMeanValues)

            lstInputFeaturesRaw = tmpRepo.getIntegerFeatures()
            lstInputFeatures = lstIntegerAttributes

            self.matIntegerTrainingData.append(tmpRepo.getNormedFeatures(self.lstMeanValues))


            if self.bUseStringFeatures:
                lstInputFeatures += tmpRepo.getWordOccurences(self.lstVoc)
                lstInputFeaturesRaw += tmpRepo.getWordOccurences(self.lstVoc)
            lstInputFeatures += tmpRepo.getRepoLanguageAsVector()
            lstInputFeaturesRaw += tmpRepo.getRepoLanguageAsVector()

            # test using unnormed features

            self.lstTrainData.append(lstInputFeatures)
            self.lstTrainDataRaw.append(lstInputFeaturesRaw)

        print("lstTrainData:")
        print(self.lstTrainData)

        print("lstTrainLabels:")
        print(self.lstTrainLabels)

        print('self.matIntegerTrainingData')
        print(self.matIntegerTrainingData)
        print('~~~~~~~~~~ NORMALIZE ~~~~~~~~~~~~~')

        self.normalizer = preprocessing.Normalizer()
        self.normalizer.fit(self.lstTrainData)

        self.normalizerIntegerAttr = preprocessing.Normalizer()
        self.normalizerIntegerAttr.fit(self.matIntegerTrainingData)

        self.lstTrainData = self.normalizer.transform(self.lstTrainData)

        return self.lstTrainData, self.lstTrainLabels

    def trainModel(self, lstTrainData, lstTrainLabels):
        """
        trains the model called self.clf with the given trainData and trainLabels

        :param lstTrainData: list
        :param lstTrainLabels:
        :return:
        """
        print('~~~~~~~~~~ TRAIN THE MODEL ~~~~~~~~~~')
        # train the nearest neighbour-model
        # "the shrink_threshold" parameter has only negative impact on the prediction results
        self.clf = NearestCentroid()

        # test out other classifiers
        # self.clf = KNeighborsClassifier()
        # self.clf = SVC()
        # self.clf = RadiusNeighborsClassifier(radius=100)
        # self.clf = MLPClassifier()
        # self.clf = GaussianProcessClassifier()
        # self.clf = LogisticRegression()

        # self.fit_transform()
        self.clf.fit(lstTrainData, lstTrainLabels)

        # this will break the machine
        # self.plotTheResult(lstTrainData, lstTrainLabels)

        self.bModelTrained = True


    def plotTheResult(self,lstTrainData, lstTrainLabels):
        """
        this is currently empty -> see the plots in the GUI instead

        :param lstTrainData: matrix which was used for training
        :param lstTrainLabels: labels which were used for training
        :return:
        """
        pass


    def exportModelToFile(self):
        """
        exports the trained model and the mean values of the input variables to './model/'
        the export is done via joblib.dump() to .pkl-file

        :return:
        """

        if self.bModelTrained:
            print('~~~~~~~~~~ SAVE MODEL TO FILE ~~~~~~~')
            # http://scikit-learn.org/stable/modules/model_persistence.html
            # http://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn

            # save the trained classifier to a file
            joblib.dump(self.clf, self.strModelPath + self.strModelFileName)
            joblib.dump(self.lstMeanValues, self.strModelPath + self.strLstMeanValuesFileName)
            joblib.dump(self.matIntegerTrainingData, self.strModelPath + self.strMatIntegerTrainingData)
            joblib.dump(self.lstTrainLabels, self.strModelPath + self.strLstTrainLabels)
            joblib.dump(self.lstTrainData, self.strModelPath + self.strLstTrainData)
            joblib.dump(self.normalizer, self.strModelPath + self.strNormalizer)
            joblib.dump(self.normalizerIntegerAttr, self.strModelPath + self.strNormalizerIntegerAttr)
            joblib.dump(self.lstTrainDataRaw, self.strModelPath + self.strLstTrainDataRaw)

    def loadModelFromFile(self):
        """
        loads / imports the model-object from './model/RepositoryClassifier.pkl'
        and the list of the mean values from './model/lstMeanValues.pkl'

        :return:
        """

        print('~~~~~~~~~~ LOAD THE MODEL ~~~~~~~~~~~')

        # load the classifier from the file
        self.clf = joblib.load(self.strModelPath + self.strModelFileName)
        self.lstMeanValues = joblib.load(self.strModelPath + self.strLstMeanValuesFileName)
        # load the integer training data for later plotting
        self.matIntegerTrainingData = joblib.load(self.strModelPath + self.strMatIntegerTrainingData)
        self.lstTrainLabels = joblib.load(self.strModelPath + self.strLstTrainLabels)
        self.lstTrainData = joblib.load(self.strModelPath + self.strLstTrainData)
        self.normalizer = joblib.load(self.strModelPath + self.strNormalizer)
        self.normalizerIntegerAttr = joblib.dump(self.normalizerIntegerAttr, self.strModelPath + self.strNormalizerIntegerAttr)
        self.lstTrainDataRaw = joblib.load(self.strModelPath + self.strLstTrainDataRaw)


        print('lstMeanValues: ', self.lstMeanValues)
        print('~~~~~~~~~~ GET THE VOCABULARY ~~~~~~~~~~')

        strVocabPath = self.directory + '/vocab/'


        strVocabPath += 'vocabList.dump'
        self.lstVoc = readVocabFromFile(strVocabPath)
        # only print out the first 7 and the last 7 entries
        # http://stackoverflow.com/questions/646644/how-to-get-last-items-of-a-list-in-python
        print('len(self.lstVoc):', len(self.lstVoc))
        if len(self.lstVoc) > 14:
            print("[", end="")
            print(*self.lstVoc[:7], sep=", ", end=" ")
            print('...', end=" ")
            print(*self.lstVoc[-7:], sep=", ", end="")
            print("]")

        self.bModelLoaded = True

        return self.clf, self.lstMeanValues, self.matIntegerTrainingData, self.lstTrainLabels, self.lstTrainData, self.normalizer, self.normalizerIntegerAttr, self.lstTrainDataRaw

    def predictResultsAndCompare(self, strProjPathFileNameCSV =  '/data/csv/manual_classification_appendix_b.csv'):  # '/data/csv/additional_data_sets_cleaned.csv'):
        """
        loads a csv-file with of layout 'URL, CATEGORY, CATEGORY_ALTERNATIVE_1,CATEGORY_ALTERNATIVE_2'
        the URL is given in the format 'https://github.com/owner/repository-name'
        the CATEGORY, CATEGORY_ALTERNATIVE_1,CATEGORY_ALTERNATIVE_2 is given by one of these options 'DEV', 'HW', 'EDU',
         'DOCS', 'WEB', 'DATA', 'OTHER'
        After the predicition phase the result is compared with the given CATEGORY and CATEGORY_ALTERNATIVES
        A verification matrix is created and the accuracy is calculated from 0.0 to 1.0

        :param strProjPathFileNameCSV: path relative to the project-path where the csv file is stored
        :return: the accuracy value (0.0 - 1.0)
        """

        if not self.bModelLoaded and not self.bModelTrained:
            print('the model hasn\'t been loaded or trained yet')
            return

        print('~~~~~~~~~~ CREATE VERITY COMP MATRIX ~~~~~~~~')

        print('~~~~~~~~~~ PREDICT RESULTS ~~~~~~~~~~')
        # classify the result

        # read the unlabeled data set from a csv
        dtUnlabeledData = pd.read_csv(self.directory + strProjPathFileNameCSV, header=0, delimiter=",")  # , nrows=iNumOfPredictions)

        # http://stackoverflow.com/questions/15943769/how-to-get-row-count-of-pandas-dataframe
        iNumOfPredictions = len(dtUnlabeledData.index)

        print('~~~~~~~~~~~ CREATE VERITY MATRIX ~~~~~~~~~~~~')
        matPredictionTarget = np.zeros((iNumOfPredictions, self.iNumCategories))

        # use a verity matrix to validate the result
        matPredictionRes = np.copy(matPredictionTarget)
        matPredictionResWithAlt = np.copy(matPredictionTarget)

        for i in range(iNumOfPredictions):

            # set the verity matrix
            strTarget = dtUnlabeledData["CATEGORY"][i]
            strTargetAlt1 = dtUnlabeledData["CATEGORY_ALTERNATIVE_1"][i]
            strTargetAlt2 = dtUnlabeledData["CATEGORY_ALTERNATIVE_2"][i]

            print('strTarget: ', strTarget)

            if pd.notnull(strTargetAlt1):
                print('strTargetAlt1:', strTargetAlt1)
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt1)] = 1

            if pd.notnull(strTargetAlt2):
                # strTargetAlt2 = strTargetAlt2[1:]
                # print('i:', i)
                print('strTargetAlt2:', strTargetAlt2)
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt2)] = 1

            iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = self.predictCategoryFromURL(dtUnlabeledData["URL"][i])

            matPredictionTarget[i, self.lstStrCategories.index(strTarget)] = 1

            print()

            matPredictionRes[i, iLabel] = 1
            matPredictionResWithAlt[i, iLabel] = 1
            matPredictionResWithAlt[i, iLabelAlt] = 1

        matPredictionResultByCategory = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # print("matPredictionTarget:", matPredictionTarget)

        for i in range(0, iNumOfPredictions):
            for j in range(0, 7):
                if matPredictionRes[i][j] == 1:
                    if matPredictionRes[i][j] == matPredictionTarget[i][j]:
                        matPredictionResultByCategory[j][0] += 1
                        matPredictionResultByCategory[j][1] += 1
                        print("i, j", i, j, matPredictionResultByCategory)
                    else:
                        matPredictionResultByCategory[j][0] += 1
                        print("i, j, not", i, j, matPredictionResultByCategory)

        for i in range (0, len(matPredictionResultByCategory)):
            matPredictionResultByCategory[i][2] = matPredictionResultByCategory[i][1] / matPredictionResultByCategory[i][0]

        self.__printResult(tmpRepo, iLabel, iLabelAlt)

        print('verity matrix for matPredictionTarget:\n ', matPredictionTarget)
        print('verity matrix for matPredictionRes:\n ', matPredictionRes)

        matCompRes = np.multiply(matPredictionTarget, matPredictionRes)
        matCompResAlt = np.multiply(matPredictionTarget, matPredictionResWithAlt)
        fPredictionRes = sum(matCompRes.flatten()) / iNumOfPredictions
        fPredictionResWithAlt = sum(matCompResAlt.flatten()) / iNumOfPredictions
        print('fPredictionRes:', fPredictionRes)
        print('fPredictionResWithAlt:', fPredictionResWithAlt)
        fAccuracy = fPredictionRes * 100
        print('fAccuracy: ', fAccuracy, '%\n')

        print('DEV: found:', matPredictionResultByCategory[0][0], ', correct:', matPredictionResultByCategory[0][1], ', reliability:', matPredictionResultByCategory[0][2] * 100, '%',
              '\nHW: found:', matPredictionResultByCategory[1][0], ', correct:', matPredictionResultByCategory[1][1], ', reliability:', matPredictionResultByCategory[1][2] * 100, '%',
              '\nEDU: found:', matPredictionResultByCategory[2][0], ', correct:', matPredictionResultByCategory[2][1], ', reliability:', matPredictionResultByCategory[2][2] * 100, '%',
              '\nDOCS: found:', matPredictionResultByCategory[3][0], ', correct:', matPredictionResultByCategory[3][1], ', reliability:', matPredictionResultByCategory[3][2] * 100, '%',
              '\nWEB: found:', matPredictionResultByCategory[4][0], ', correct:', matPredictionResultByCategory[4][1], ', reliability:', matPredictionResultByCategory[4][2] * 100, '%',
              '\nDATA: found:', matPredictionResultByCategory[5][0], ', correct:', matPredictionResultByCategory[5][1], ', reliability:', matPredictionResultByCategory[5][2] * 100, '%',
              '\nOTHER: found:', matPredictionResultByCategory[6][0], ', correct:', matPredictionResultByCategory[6][1], ', reliability:', matPredictionResultByCategory[6][2] * 100, '%')

        return fPredictionRes

    def predictCategoryFromOwnerRepoName(self, strUser, strRepoName):
        """
        predicts the category for a repository which is given by the user and repo-name

        :param strUser: owner of the repository
        :param strRepoName: name of the repository
        :return:
        """
        tmpRepo = GithubRepo(strUser, strRepoName)

        return self.predictCategoryFromGitHubRepoObj(tmpRepo)

    def predictCategoryFromURL(self, strGitHubRepoURL):
        """
        loads the features of a given repository by URL and the model predicts its category-label

        :param strGitHubRepoURL: url to the repository
        :return: label value form 0 - 6, lst of the precentages for the other categories
        """
        try:
            tmpRepo = GithubRepo.fromURL(strGitHubRepoURL)
        except Exception as ex:
            raise ex

        return self.predictCategoryFromGitHubRepoObj(tmpRepo)

    def predictCategoryFromGitHubRepoObj(self, tmpRepo):
        """
        predicts the category for a GithubRepo-Object
        :param tmpRepo: GithubRepo-Object
        :return: iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures
        """

        lstNormedInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues)
        if self.bUseStringFeatures:
            lstNormedInputFeatures += tmpRepo.getWordOccurences(self.lstVoc)
        lstNormedInputFeatures += tmpRepo.getRepoLanguageAsVector()

        # apply pre-processing
        lstNormedInputFeatures = np.array(lstNormedInputFeatures).reshape(1, len(lstNormedInputFeatures))

        lstNormedInputFeatures = self.normalizer.transform(lstNormedInputFeatures)

        # reshape Input Features -> otherwise a deprecation warning occurs
        iLabel = int(self.clf.predict(lstNormedInputFeatures))

        if self.bUseCentroids is True:
            matCentroids = self.clf.centroids_
            lstFinalPercentages = self.predictProbaNearestCentroids(matCentroids, lstNormedInputFeatures)
        else:
            lstFinalPercentages = self.clf.predict_proba(lstNormedInputFeatures)

        iLabelAlt = self.getLabelAlternative(lstFinalPercentages)

        self.__printResult(tmpRepo, iLabel, iLabelAlt, bPrintWordHits=False)

        return iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures

    def getLabelAlternative(self, lstFinalPercentages):
        """
        gets the first alternative (the seoond result)

        :param lstFinalPercentages: percentages lsit for the single categories
        :return: integer label which describes the category
        """
        # copy the percentages in an additional list
        lstFinalPercentagesCopy = []
        lstFinalPercentagesCopy = lstFinalPercentages[:]

        # get the s
        iMaxIndex = lstFinalPercentagesCopy.index(max(lstFinalPercentagesCopy))

        lstFinalPercentagesCopy[iMaxIndex] = 0
        iSecondMaxIndex = lstFinalPercentagesCopy.index(max(lstFinalPercentagesCopy))

        return iSecondMaxIndex


    def predictProbaNearestCentroids(self, matCentroids, lstInputFeatures):
        """
        because predictProba was missing in the default functionality for nearest-centroid
        the probability is now calculated via the distances to the different centroids

        :param matCentroids: matrix of the centroids for each category
        :param lstInputFeatures: full normed input feature list for which the prediction is based on
        :return:
        """
        lstFinalPercentages = []
        fDistSum = 0
        lstDistances = []

        for i, centroid in enumerate(matCentroids):
            fDist = np.linalg.norm([lstInputFeatures] - centroid)
            lstDistances.append((i, fDist))
            fDistSum += fDist

        lstDistances.sort(key=lambda x: x[1])

        lstPercentages = []

        for i, fDist in enumerate(lstDistances):
            lstPercentages.append(lstDistances[i][1] / fDistSum)

        lstDistancesReordered = []

        for i, fPercentage in enumerate(reversed(lstPercentages)):
            lstDistancesReordered.append((lstDistances[i][0], fPercentage))

        lstDistancesReordered.sort(key=lambda x: x[0])

        for i, fPercentage in enumerate(lstDistancesReordered):
            lstFinalPercentages.append(fPercentage[1])
            print('{:15s} {:3f}'.format(self.lstStrCategories[i],  fPercentage[1]))

        return lstFinalPercentages

    def __printResult(self, tmpRepo, iLabel, iLabelAlt, bPrintWordHits=False):
        """
        prints the repository name and its category by using the iLabel

        :param tmpRepo: given repository
        :param iLabel: previously predicted label
        :return:
        """

        strStopper1 = "=" * 80
        strStopper2 = "-" * 80

        print(strStopper1)
        if bPrintWordHits is True:
            if self.bUseStringFeatures:
                lstOccurence = tmpRepo.getWordOccurences(self.lstVoc)
                tmpRepo.printFeatureOccurences(tmpRepo.getDicFoundWords())
                # printFeatureOccurences(self.lstVoc, lstOccurence, 0)


        print('Prediction for ' + tmpRepo.getName() + ', ' + tmpRepo.getUser() + ': ', end="")

        print(self.lstStrCategories[iLabel])
        if iLabelAlt is not None:
            print('Alternative: ', self.lstStrCategories[iLabelAlt])

        print(strStopper2)
