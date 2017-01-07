from operator import add
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors.nearest_centroid import NearestCentroid

from githubRepo import GithubRepo
from utility_funcs.count_vectorizer_operations import *
from utility_funcs.preprocessing_operations import *


#logging.basicConfig(level=logging.DEBUG)

class RepoClassifierNearestNeighbour:

    def __init__(self):
        """
        constructor which initializes member variables

        """
        self.bModelLoaded = False
        self.bModelTrained = False
        self.clf = None
        self.lstMeanValues = None
        self.lstVoc = None

        self.lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']
        
        self.directory = path.dirname(__file__)
        print(self.directory)

        # get the project-directory
        self.strProjectDir = str(Path().resolve().parent)
        print('strProjectDir:', self.strProjectDir)

        self.strModelPath = self.directory + '/model/'

        # Create model-directory if needed
        if not os.path.exists(self.strModelPath):
            os.makedirs(self.strModelPath)
        self.strModelFileName = 'RepoClassifierNearestNeighbour.pkl'
        self.strLstMeanValuesFileName = 'lstMeanValues.pkl'

        self.iNumCategories = len(self.lstStrCategories)

    def trainModel(self, strProjPathFileNameCSV ='/data/csv/additional_data_sets_cleaned.csv'):
        """
        trains the model with a given csv-file. the csv file must have 2 columns URL and CATEGORY.
        the URL is given in the form 'https://github.com/owner/repository-name'
        the CATEGORY is given by one of these options 'DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER'

        :param strProjPathFileNameCSV: file path relative to the project-path where the csv-file is stored
        :return:
        """

        # trainData = pd.read_csv(directory + "/example_repos.csv", header=0, delimiter=",",
        trainData = pd.read_csv(self.strProjectDir + strProjPathFileNameCSV, header=0, delimiter=",")

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
        for tmpGithubRepo in lstGithubRepo:
            # lstMeanValues += tmpGithubRepo.getFeatures()
            self.lstMeanValues = list(map(add, self.lstMeanValues, tmpGithubRepo.getFeatures()))

            # find the according label as an intger for the current repository
            # the label is defined in trainData
            lstTrainLabels.append(self.lstStrCategories.index(trainData["CATEGORY"][i]))
            i += 1

        # replace every 0 with 1, otherwise division by 0 occurs
        # http://stackoverflow.com/questions/2582138/finding-and-replacing-elements-in-a-list-python
        # self.lstMeanValues[:] = [1 if x == 0 else x for x in self.lstMeanValues]

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
            lstInputFeatures = tmpGithubRepo.getNormedFeatures(self.lstMeanValues) + tmpGithubRepo.getWordOccurences(self.lstVoc)

            # test using unnormed features
            # lstInputFeatures = tmpGithubRepo.getFeatures() + tmpGithubRepo.getWordOccurences(lstVoc)

            lstTrainData.append(lstInputFeatures)

        print("lstTrainData:")
        print(lstTrainData)

        print("lstTrainLabels:")
        print(lstTrainLabels)

        print('~~~~~~~~~~ NORMALIZE ~~~~~~~~~~~~~')
        # lstTrainData = preprocessing.normalize(lstTrainData)

        # print("lstTrainDataNormalized:", lstTrainData)

        print('~~~~~~~~~~ TRAIN THE MODEL ~~~~~~~~~~')
        # train the nearest neighbour-model
        self.clf = NearestCentroid()
        self.clf.fit(lstTrainData, lstTrainLabels)

        self.bModelTrained = True


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
        loads / imports the model-object from './model/RepoClassifierNearestNeighbour.pkl'
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
        # unlabeledData = pd.read_csv(strProjectDir + '/data/csv/unclassified_repos.csv', header=0, delimiter=",")#, nrows=iNumOfPredictions)
        dtUnlabeledData = pd.read_csv(self.strProjectDir + strProjPathFileNameCSV, header=0,
                                      delimiter=",")  # , nrows=iNumOfPredictions)

        # http://stackoverflow.com/questions/15943769/how-to-get-row-count-of-pandas-dataframe
        # len(dtFrame.index) gets the number of rows NaN are included
        iNumOfPredictions = len(dtUnlabeledData.index)


        print('~~~~~~~~~~~ CREATE VERITY MATRIX ~~~~~~~~~~~~')
        matPredictionTarget = np.zeros((iNumOfPredictions, self.iNumCategories))

        # use a verity matrix to validate the result
        matPredictionRes = np.copy(matPredictionTarget)

        for i in range(iNumOfPredictions):
            tmpRepo = GithubRepo.fromURL(dtUnlabeledData["URL"][i])

            lstInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues) + tmpRepo.getWordOccurences(self.lstVoc)

            # iLabel = int(clf.predict([tmpRepo.getNormedFeatures(lstMeanValues)]))
            iLabel = int(self.clf.predict([lstInputFeatures]))
            # set the verity matrix
            strTarget = dtUnlabeledData["CATEGORY"][i]
            strTargetAlt1 = dtUnlabeledData["CATEGORY_ALTERNATIVE_1"][i]
            strTargetAlt2 = dtUnlabeledData["CATEGORY_ALTERNATIVE_2"][i]

            matPredictionTarget[i, self.lstStrCategories.index(strTarget)] = 1

            if pd.notnull(strTargetAlt1):
                # strTargetAlt1 = strTargetAlt1[1:]
                # print('i:', i)
                print('strTargetAlt1:', strTargetAlt1)
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt1)] = 1

            if pd.notnull(strTargetAlt2):
                # strTargetAlt2 = strTargetAlt2[1:]
                # print('i:', i)
                # print('strTargetAlt2:', strTargetAlt2)
                matPredictionTarget[i, self.lstStrCategories.index(strTargetAlt2)] = 1

            matPredictionRes[i, iLabel] = 1

            lstOccurence = tmpRepo.getWordOccurences(self.lstVoc)
            # print('lstOccurence:', lstOccurence)
            printFeatureOccurences(self.lstVoc, lstOccurence, 0)
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

    def predictCategoryFromURL(self, strGitHubRepoURL):
        """
        loads the features of a given repository by URL and the model predicts its category-label

        :param strGitHubRepoURL: url to the repository
        :return: label value form 0 - 6
        """
        tmpRepo = GithubRepo.fromURL(strGitHubRepoURL)

        lstInputFeatures = tmpRepo.getNormedFeatures(self.lstMeanValues) + tmpRepo.getWordOccurences(self.lstVoc)

        iLabel = int(self.clf.predict([lstInputFeatures]))

        self.__printResult(tmpRepo, iLabel)

        return iLabel

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
        print('Prediction for ' + tmpRepo.getName() + ', ' + tmpRepo.getUser() + ': ', end="")

        print(self.lstStrCategories[iLabel])
        print(strStopper2)
