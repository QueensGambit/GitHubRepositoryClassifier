import requests
import features.learning_features
from sklearn.feature_extraction.text import CountVectorizer

import datetime
from os import path
import os
import json

import utility_funcs.string_operation
import utility_funcs.count_vectorizer_operations

from utility_funcs.io_agent import InputOutputAgent

import numpy as np
from features.learning_features import IntFeatures, StringFeatures
import definitions.githubLanguages

# http://stackoverflow.com/questions/32910096/is-there-a-way-to-auto-generate-a-str-implementation-in-python
def auto_str(cls):
    """
    method for auto-generating a to string-function which prints out all member-attributes

    :param cls: current class
    :return: cls
    """
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            # ', '.join('%s=%s' % item for item in vars(self).items())
            '\n '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls


@auto_str
class GithubRepo:

    def __init__(self, strUser, strName):
        """
        Simple constructor

        :param strUser: user of the repository
        :param strName: name of the repository
        """
        self.user = strUser
        self.name = strName
        # print('user: ', self.user, 'name: ', self.name)

        d = path.dirname(__file__)
        self.strPathJSON = d + '/json/' + strUser + '_' + strName + '.json'

        self.ioAgent = InputOutputAgent(strUser, strName, bWithToken=True)

        self.apiJSON, self.apiUrl, self.lstReadmePath = self.ioAgent.loadJSONdata(self.strPathJSON)

        self.strDirPath_readme = os.path.abspath(os.path.join(__file__, os.pardir)) + '\\readme'
        # print(self.ioAgent.getReadme(self.strDirPath_readme))


        self.intFeatures = None
        self.strFeatures = None

        print('url: ' + str(self.apiUrl))
        self.readAttributes()


    def getFilteredRepoDescription(self):
        strDescription = self.apiJSON['description']
        if strDescription:
            # return strDescription
            return string_operation.prepare_words(strDescription)
        else:
            return ""

    def getRepoLanguage(self):
        strLanguage = self.apiJSON['language']
        # return string_operation.prepare_words(strLanguage)
        if strLanguage:
            return strLanguage
        else:
            return "undetected"

    def getRepoLanguageAsVector(self):
        lstLangVec = [0] * len(definitions.githubLanguages.lstLanguages)
        try:
            iLangIndex = definitions.githubLanguages.lstLanguages.index(self.getRepoLanguage())
        except ValueError:
            iLangIndex = definitions.githubLanguages.lstLanguages.index("rare")

        lstLangVec[iLangIndex] = 1 #1

        return lstLangVec


    def getFilteredReadme(self):
        """
        returns the filtered readme with prepare_words() being applied

        :return: string of the filtered readme
        """
        strMyREADME = self.ioAgent.getReadme(self.strDirPath_readme)
        return string_operation.prepare_words(strMyREADME)

    @classmethod
    def fromURL(cls, strURL):
        """
        constructor with url instead of user, name

        :param strURL: url of the github-repository
        :return: calls the main-constructor
        """
        iIndexUser = 3
        iIndexName = 4
        lststrLabelGroup = strURL.split('/')
        return cls(lststrLabelGroup[iIndexUser], lststrLabelGroup[iIndexName])


    def readAttributes(self):
        """
        reads all attributes of the json-file and fills the integer-attributes

        :return:
        """
        # print('readAttributes...')

        # strUrl = 'https://api.github.com/repos/WebpageFX/emoji-cheat-sheet.com'

        # self.apiJSON = requests.get(strUrl)
        # self.apiJSON = requests.get(self.apiUrl)

        # a usual Github-Time stamp looks like this:
        # "2011-10-17T15:09:52Z"

        # example conversion: stackoverflow.com/questions/5385238/python-reading-logfile-with-timestamp-including-microseconds
        # >>> s = "2010-01-01 18:48:14.631829"
        # >>> datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")

        strGithubTimestampFormat = "%Y-%m-%dT%H:%M:%SZ"
        datStart = datetime.datetime.strptime(self.apiJSON['created_at'], strGithubTimestampFormat)
        # last update is a push or change in wiki, description...
        datLastUpdate = datetime.datetime.strptime(self.apiJSON['updated_at'], strGithubTimestampFormat)
        iDevTime = (datLastUpdate - datStart).days

        # print('iDevTime:', iDevTime)

        # jsBranches = self.apiJSON['branches_url'])).json()
        # iNumBranches = len(jsBranches)

        self.intFeatures = IntFeatures(iSubscriberCount=self.apiJSON['subscribers_count'],
                                       iOpenIssues=self.apiJSON['open_issues'],
                                       iDevTime=iDevTime,
                                       dRepoActivity=0,
                                       dCommitIntervals=0,
                                       iWatchersCount=0, #self.apiJSON['watchers_count'],
                                       iSize=self.apiJSON['size'])

        # print(self.apiJSON['contributors_url'])
        # jsContrib = (requests.get(self.apiJSON['contributors_url'])).json()

        # print('len(jsContrib):', len(jsContrib)) # better use subscriber-count ther contributor length only lists the top contributors


    def getFeatures(self):
        """
        gets the intFeatures as a list

        :return: list of the integer features
        """
        lstFeatures = [self.intFeatures.iSubscriberCount,
                       self.intFeatures.iOpenIssues,
                       self.intFeatures.iDevTime,
                       self.intFeatures.dRepoActivity, #dCodeFrequency
                       self.intFeatures.dCommitIntervals,
                       self.intFeatures.iWatchersCount,  #iNumBranches
                       self.intFeatures.iSize
                       ]

        # skip int features
        # lstFeatures = [0] * 7

        return lstFeatures

    def getNormedFeatures(self, lstMeanValues):
        """
        returns the features which were normed by dividing them with the mean values

        :param lstMeanValues: mean value of every integer feature
        :return: list of the normed integer features
        """
        lstNormedFeatures = self.getFeatures()
        # norm every integer feature by dividing it with it's mean value
        # avoid dividing by 0
        lstNormedFeatures[:] = [x / y if y != 0 else 0 for x, y in zip(lstNormedFeatures, lstMeanValues)]
        return lstNormedFeatures

    def getWordOccurences(self, lstVocab):
        """
        calculates the number of occurrences of the words given by the vocab list;
        afterwards this list is divided by the word-length of the readme and multiplied with a factor

        :param lstVocab: vocabulary which is used in the CountVectorizer of scikit-learn
        :return: integer list representing the percentage-usage of the vocabulary words
        """
        # test skipping word occurrences completly in the evaluation
        return [0] * len(lstVocab)

        vectorizer = CountVectorizer(min_df=0.5, vocabulary=lstVocab)

        strFilteredReadme = ""
        # strFilteredReadme = self.getFilteredReadme()
        getFilteredRepoDescription = self.getFilteredRepoDescription()
        strFilteredReadme += getFilteredRepoDescription
        # strFilteredReadme += getFilteredRepoDescription

        # print(strFilteredReadme)

        # return a sparse matrix
        # each column is mapped to a specific feature (see lstFeatureNames)
        # the value describes the occurrence of the word in the current line
        matSparse = vectorizer.fit_transform(strFilteredReadme.split())

        lstFeatureNames = vectorizer.get_feature_names()

        # print('~~~~~~~~~~ Number-of-total-occurrences ~~~~~~~~~~')
        # print('--> repository: ' + self.user + '_' + self.name + '~~~~~~~~~~')
        matOccurrence = np.asarray(np.sum(matSparse, axis=0))

        # flatten makes a matrix 1 dimensional
        lstOccurrence = np.array(matOccurrence.flatten()).tolist()    # np.array().tolist() is not needed

        iHits = np.sum(lstOccurrence)

        # divide each element by a factor to reduce the effectiveness
        iLen = len(strFilteredReadme.split())

        # avoid dividing by 0
        if iLen == 0:
            iLen = 1

        if iHits == 0:
            iHits = 1

        # fFacEffectiveness = 10.0
        # fFacEffectiveness = 20.0

        # 10 is the factor between string and integer attributes
        # lstOccurrence[:] = [(x / iLen) * fFacEffectiveness for x in lstOccurrence]

        # keep as is
        # lstOccurrence[:] = [x for x in lstOccurrence]

        # make a binarized vector
        lstOccurrence[:] = [1 if x > 0 else 0 for x in lstOccurrence]

        # count_vectorizer_operations.printFeatureOccurences(lstFeatureNames, lstOccurrence, 2)

        return lstOccurrence


    def getName(self):
        """
        getter method for name

        :return: self.name
        """
        return self.name

    def getUser(self):
        """
        getter method for user

        :return: self.user
        """
        return self.user
