from operator import add
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors.nearest_centroid import NearestCentroid

from os import path

from .utility_funcs.count_vectorizer_operations import printFeatureOccurences
from .utility_funcs.preprocessing_operations import initInputParameters, readVocabFromFile

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

class RepositoryClassifier:

    def __init__(self, bUseStringFeatures=True):
        """
        constructor which initializes member variables

        """

        # if bOverloadPrintFunc:
            # import print_overloading

        self.bModelLoaded = False
        self.bModelTrained = False
        self.clf = None
        self.lstMeanValues = None
        self.lstVoc = None
        self.stdScaler = None

        self.lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']

        self.directory = path.dirname(__file__)
        print(self.directory)

        self.bUseStringFeatures = bUseStringFeatures

        # get the project-directory
        # self.strProjectDir = str(Path().resolve().parent.parent)
        self.strProjectDir = str(Path().resolve().parent)

        print('strProjectDir:', self.strProjectDir)

        self.strModelPath = self.directory + '/model/'

        # Create model-directory if needed
        if not os.path.exists(self.strModelPath):
            os.makedirs(self.strModelPath)
        self.strModelFileName = 'RepositoryClassifier.pkl'
        self.strLstMeanValuesFileName = 'lstMeanValues.pkl'

        self.iNumCategories = len(self.lstStrCategories)

    def loadTrainingData(self, strProjPathFileNameCSV ='/data/csv/additional_data_sets_cleaned.csv'):
        """
        trains the model with a given csv-file. the csv file must have 2 columns URL and CATEGORY.
        the URL is given in the form 'https://github.com/owner/repository-name'
        the CATEGORY is given by one of these options 'DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER'

        :param strProjPathFileNameCSV: file path relative to the project-path where the csv-file is stored
        :return:
        """

        # trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",",
        trainData = pd.read_csv(self.strProjectDir + strProjPathFileNameCSV, header=0, delimiter=",")
        # trainData = pd.read_csv(self.directory + strProjPathFileNameCSV, header=0, delimiter=",")

        # len(trainData.index) gets the number of rows
        iNumTrainData = len(trainData.index)
        print("iNumTrainData: ", iNumTrainData)

        print('~~~~~~~~~~ EXTRACTING FEATURES ~~~~~~~~~~')

        lstGithubRepo = []

        for i in range(iNumTrainData):
            # fill the list with GithubRepo-Objects
            # print('string_operations.extractProjectNameUser: ', string_operations.extractProjectNameUser(trainData["URL"][i]))
            lstGithubRepo.append(GithubRepo.fromURL(trainData["URL"][i]))

        # fill the train and the label-data
        lstTrainData = []
        lstTrainLabels = []

        print('~~~~~~~~~~ CALCULATE THE MEAN VALUES ~~~~~~~~~~')
        self.lstMeanValues = [0] * 7
        i = 0
        for tmpRepo in lstGithubRepo:
            # print out the description
            # print('descr:', tmpGithubRepo.getFilteredRepoDescription())
            # print('lang:', tmpGithubRepo.getRepoLanguage())

            # lstMeanValues += tmpGithubRepo.getIntegerFeatures()
            self.lstMeanValues = list(map(add, self.lstMeanValues, tmpRepo.getIntegerFeatures()))

            # find the according label as an intger for the current repository
            # the label is defined in trainData
            lstTrainLabels.append(self.lstStrCategories.index(trainData["CATEGORY"][i]))
            i += 1

        # replace every 0 with 1, otherwise division by 0 occurs
        # http://stackoverflow.com/questions/2582138/finding-and-replacing-elements-in-a-list-python
        # self.lstMeanValues[:] = [1 if x == 0 else x for x in self.lstMeanValues]

        # Divide each element with the number of training data
        self.lstMeanValues[:] = [x / iNumTrainData for x in self.lstMeanValues]

        # set all mean values to 1
        # self.lstMeanValues[:] = [1 for x in self.lstMeanValues]

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
        for tmpRepo in lstGithubRepo:
            # fill the Training-Data
            # ordinary integer-attributes
            # lstTrainData.append(tmpGithubRepo.getNormedFeatures(lstMeanValues))

            # lstInputFeatures = tmpGithubRepo.getNormedFeatures(lstMeanValues)
            # with the word occurrence vector
            # lstInputFeatures = lstInputFeatures + (tmpGithubRepo.getWordSparseMatrix(lstVoc))
            # print(tmpGithubRepo.getNormedFeatures(lstMeanValues))
            # print(tmpGithubRepo.getWordSparseMatrix(lstVoc))

            # np.vstack  concates to numpy-arrays
            lstInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues)
            if self.bUseStringFeatures:
                lstInputFeatures += tmpRepo.getWordOccurences(self.lstVoc)
            lstInputFeatures += tmpRepo.getRepoLanguageAsVector()


            # test using unnormed features
            # lstInputFeatures = tmpGithubRepo.getIntegerFeatures() + tmpGithubRepo.getWordOccurences(lstVoc)

            lstTrainData.append(lstInputFeatures)

        print("lstTrainData:")
        print(lstTrainData)

        print("lstTrainLabels:")
        print(lstTrainLabels)

        print('~~~~~~~~~~ NORMALIZE ~~~~~~~~~~~~~')
        # self.stdScaler = preprocessing.StandardScaler()
        # self.stdScaler.fit(lstTrainData)
        # print('stdScaler.mean:', self.stdScaler.mean_, 'stdScaler.scale:', self.stdScaler.scale_)
        #
        # lstTrainData = self.stdScaler.fit_transform(lstTrainData)
        # lstTrainData = preprocessing.normalize(lstTrainData)

        # print("lstTrainDataNormalized:", lstTrainData)

        return lstTrainData, lstTrainLabels

    def trainModel(self, lstTrainData, lstTrainLabels):
        print('~~~~~~~~~~ TRAIN THE MODEL ~~~~~~~~~~')
        # train the nearest neighbour-model
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
        reduced_data = PCA(n_components=2).fit_transform(lstTrainData)
        kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

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
        print('lstMeanValues: ', self.lstMeanValues)

        print('~~~~~~~~~~ GET THE VOCABULARY ~~~~~~~~~~')
        strVocabPath = self.directory + '/vocab/'

        strVocabPath += 'vocabList.dump'
        self.lstVoc = readVocabFromFile(strVocabPath)

        self.bModelLoaded = True

    def predictResultsAndCompare(self, strProjPathFileNameCSV = '/data/csv/manual_classification_appendix_b.csv'):
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

        # as a sample prediction example use an array of 42, 42, ... as an example feature set
        # iNumbeTrainingFeatures = len(lstGithubRepo[0].getNormedFeatures(lstMeanValues)) + len(lstVoc)
        # iLabel = int(clf.predict([[42] * iNumbeTrainingFeatures]))
        #
        # print('iLabel:', iLabel)
        # print('Prediction for 42,42:', lstStrCategories[iLabel])

        # iNumOfPredictions = 7
        # read the unlabeled data set from a csv
        dtUnlabeledData = pd.read_csv(self.strProjectDir + strProjPathFileNameCSV, header=0, delimiter=",")  # , nrows=iNumOfPredictions)

        # http://stackoverflow.com/questions/15943769/how-to-get-row-count-of-pandas-dataframe
        # len(dtFrame.index) gets the number of rows NaN are included
        iNumOfPredictions = len(dtUnlabeledData.index)


        print('~~~~~~~~~~~ CREATE VERITY MATRIX ~~~~~~~~~~~~')
        matPredictionTarget = np.zeros((iNumOfPredictions, self.iNumCategories))

        # use a verity matrix to validate the result
        matPredictionRes = np.copy(matPredictionTarget)

        for i in range(iNumOfPredictions):
            tmpRepo = GithubRepo.fromURL(dtUnlabeledData["URL"][i])

            lstInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues)
            if self.bUseStringFeatures:
                lstInputFeatures += tmpRepo.getWordOccurences(self.lstVoc)
            lstInputFeatures += tmpRepo.getRepoLanguageAsVector()

            # apply pre-processing
            # lstInputFeatures = self.stdScaler.fit_transform(lstInputFeatures)

            # iLabel = int(clf.predict([tmpRepo.getNormedFeatures(lstMeanValues)]))
            iLabel = int(self.clf.predict([lstInputFeatures]))
            # set the verity matrix
            strTarget = dtUnlabeledData["CATEGORY"][i]
            strTargetAlt1 = dtUnlabeledData["CATEGORY_ALTERNATIVE_1"][i]
            strTargetAlt2 = dtUnlabeledData["CATEGORY_ALTERNATIVE_2"][i]

            matPredictionTarget[i, self.lstStrCategories.index(strTarget)] = 1

            print('strTarget: ', strTarget, end="")
            if pd.notnull(strTargetAlt1):
                # strTargetAlt1 = strTargetAlt1[1:]
                # print('i:', i)
                print('strTargetAlt1:', strTargetAlt1, end="")
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt1)] = 1

            if pd.notnull(strTargetAlt2):
                # strTargetAlt2 = strTargetAlt2[1:]
                # print('i:', i)
                print('strTargetAlt2:', strTargetAlt2, end="")
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt2)] = 1

            print()

            matPredictionRes[i, iLabel] = 1

            # print('len(lstOccurence):', len(lstOccurence))

            self.__printResult(tmpRepo, iLabel)

        print('verity matrix for matPredictionTarget:\n ', matPredictionTarget)
        print('verity matrix for matPredictionRes:\n ', matPredictionRes)

        matCompRes = np.multiply(matPredictionTarget, matPredictionRes)
        fPredictionRes = sum(matCompRes.flatten()) / iNumOfPredictions
        print('fPredictionRes:', fPredictionRes)
        fAccuracy = fPredictionRes * 100
        print('fAccuracy: ', fAccuracy, '%')

        return fPredictionRes

    def predictCategoryFromOwnerRepoName(self, strUser, strRepoName):
        tmpRepo = GithubRepo(strUser, strRepoName)

        return self.predictCategoryFromGitHubRepoObj(tmpRepo)

    def predictCategoryFromURL(self, strGitHubRepoURL):
        """
        loads the features of a given repository by URL and the model predicts its category-label

        :param strGitHubRepoURL: url to the repository
        :return: label value form 0 - 6, lst of the precentages for the other categories
        """
        tmpRepo = GithubRepo.fromURL(strGitHubRepoURL)

        return self.predictCategoryFromGitHubRepoObj(tmpRepo)

    def predictCategoryFromGitHubRepoObj(self, tmpRepo):

        lstInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues)
        if self.bUseStringFeatures:
            lstInputFeatures += tmpRepo.getWordOccurences(self.lstVoc)
        lstInputFeatures += tmpRepo.getRepoLanguageAsVector()

        # apply pre-processing
        # lstInputFeatures = self.stdScaler.fit_transform(lstInputFeatures)

        iLabel = int(self.clf.predict([lstInputFeatures]))

        # res = self.clf.predict_proba([lstInputFeatures])
        # print('self.clf.predict_proba()', res)
        # score = self.clf.score(self.clf.predict([lstInputFeatures]), [0] * 7)
        # print('score:', score)

        matCentroids = self.clf.centroids_

        fDistSum = 0
        lstDistances = []

        for i, centroid in enumerate(matCentroids):
            # print(centroid)
            fDist = np.linalg.norm([lstInputFeatures] - centroid)
            lstDistances.append((i, fDist))
            # print('fDist:', fDist)
            fDistSum += fDist

        lstDistances.sort(key=lambda x: x[1])

        # print('sorted:', lstDistances)

        lstPercentages = []

        for i, fDist in enumerate(lstDistances):
            lstPercentages.append(lstDistances[i][1] / fDistSum)

        lstDistancesReordered = []

        for i, fPercentage in enumerate(reversed(lstPercentages)):
            lstDistancesReordered.append((lstDistances[i][0], fPercentage))

        lstDistancesReordered.sort(key=lambda x: x[0])

        lstFinalPercentages = []

        for i, fPercentage in enumerate(lstDistancesReordered):
            lstFinalPercentages.append(fPercentage[1])
            print(self.lstStrCategories[i], 'pecentage:', fPercentage[1])

        # print(self.clf.centroids_)

        self.__printResult(tmpRepo, iLabel)

        return iLabel, lstFinalPercentages


    def __printResult(self, tmpRepo, iLabel):
        """
        prints the repository name and its category by using the iLabel

        :param tmpRepo: given repository
        :param iLabel: previously predicted label
        :return:
        """

        strStopper1 = "=" * 80
        strStopper2 = "-" * 80

        print(strStopper1)
        if self.bUseStringFeatures:
            lstOccurence = tmpRepo.getWordOccurences(self.lstVoc)
            printFeatureOccurences(self.lstVoc, lstOccurence, 0)

        print('Prediction for ' + tmpRepo.getName() + ', ' + tmpRepo.getUser() + ': ', end="")

        print(self.lstStrCategories[iLabel])
        print(strStopper2)
