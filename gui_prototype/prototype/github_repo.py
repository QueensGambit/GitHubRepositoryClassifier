import datetime
import os
from os import path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .definitions.githubLanguages import lstLanguages
from .features.learning_features import IntFeatures
from .utility_funcs import string_operation
from .utility_funcs.io_agent import InputOutputAgent
from pathlib import Path
# http://stackoverflow.com/questions/32910096/is-there-a-way-to-auto-generate-a-str-implementation-in-python
def auto_str(cls):
    """
    Method for auto-generating a to string-function which prints out all member-attributes

    :param cls: current class
    :return: cls
    """
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            '\n '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls


@auto_str
class GithubRepo:

    def __init__(self, strUser, strName):
        """
        Constructor which takes two arguments to initialize the repository

        :param strUser: user of the repository (e.g. "GNOME")
        :param strName: name of the repository (e.g. "gimp")
        """
        self.user = strUser
        self.name = strName
        # print('user: ', self.user, 'name: ', self.name)

        d = path.dirname(__file__)
        # d = str(Path())

        self.strPathJSON = d + '/json/' + strUser + '_' + strName + '.json'

        try:
            self.ioAgent = InputOutputAgent(strUser, strName)
            self.apiJSON, self.apiUrl, self.lstReadmePath = self.ioAgent.loadJSONdata(self.strPathJSON)

            self.strDirPath_readme = os.path.abspath(os.path.join(__file__, os.pardir)) + '\\readme'

            self.intFeatures = None
            self.strFeatures = None
            self.lstOccurrence = None
            self.strFilteredReadme = None
            self.dicFoundWords = None

            print('url: ' + str(self.apiUrl))
            self.readAttributes()
        except ConnectionError as e:
            raise e


    def getRepoDescription(self):
        """
        Gets the full description of the repository which is stored in the json-Api
        If the description wasn't set, an empty string "" will be returned

        :return: string which contains the description
        """
        strDescr = self.apiJSON['description']
        if strDescr is None:
            return ""
        else:
            return strDescr

    def getFilteredRepoDescription(self, bApplyStemmer=True, bCheckStopWords=False):
        """
        gets a filtered version of the description of the repository
        if the description wasn't set, an empty string "" will be returned

        :param bApplyStemmer: true if the words should be stripped to the stem
        :param bCheckStopWords: true if known stopwords such as (the, he, and,...) should be ignored
        :return: string which contains the filtered form of the description
        """
        strDescription = self.getRepoDescription()
        if strDescription is not "":
            # return strDescription
            return string_operation.prepare_words(strDescription, bApplyStemmer, bCheckStopWords)
        else:
            return ""

    def getRepoLanguage(self):
        """
        Gets the language from the main json-Api-page which was assigned by github to this repository.
        If no language was allocated "undetected" will be returned

        :return: string which contains the language (e.g. C++, Java, Python,...)
        """
        strLanguage = self.apiJSON['language']
        if strLanguage is not None:
            return strLanguage
        else:
            return "undetected"

    def getRepoLanguageAsVector(self):
        """
        Returns an integer-list with 102 entries
        All of them are set to 0 except the language which is used

        :return: list
        """
        lstLangVec = [0] * len(lstLanguages)
        try:
            iLangIndex = lstLanguages.index(self.getRepoLanguage())
        except ValueError:
            iLangIndex = lstLanguages.index("rare")

        lstLangVec[iLangIndex] = 1

        return lstLangVec


    def getReadme(self):
        """
        Gets the raw content of the readme of the repository which can either be a README.md or README.rst file.
        The job for loading and exporting the readme is done by it's Io-Agent.

        :return: string with the raw content
        """
        strMyREADME = self.ioAgent.getReadme(self.strDirPath_readme)
        return strMyREADME

    def getFilteredReadme(self, bApplyStemmer=True, bCheckStopWords=False):
        """
        Returns the filtered readme with prepare_words() being applied

        :return: string of the filtered readme
        """
        if self.strFilteredReadme is None:
            self.strFilteredReadme = string_operation.prepare_words(self.getReadme(), bApplyStemmer, bCheckStopWords)

        return self.strFilteredReadme

    def getDevTime(self):
        """
        Gets the devolpment time of the repository in days.
        This is calculated via the difference of 'created_at' - 'updated_at'

        :return: integer which
        """
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

        return iDevTime

    def getNumOpenIssue(self):
        """
        gets the number of open issues from the json-main-page

        :return:
        """
        return self.apiJSON['open_issues']

    def getNumWatchers(self):
        """
        gets the number of watcher from the json-main-page
        :return:
        """
        return self.apiJSON['watchers_count']

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
        iDevTime = self.getDevTime()

        self.intFeatures = IntFeatures(iSubscriberCount=self.apiJSON['subscribers_count'],
                                       iOpenIssues=self.getNumOpenIssue(),
                                       iDevTime=iDevTime,
                                       iSize=self.apiJSON['size'])


    def getIntegerFeatures(self):
        """
        gets the intFeatures as a list

        :return: list of the integer features
        """
        lstFeatures = [self.intFeatures.iSubscriberCount,
                       self.intFeatures.iOpenIssues,
                       self.intFeatures.iDevTime,
                       self.intFeatures.iSize
                       ]

        return lstFeatures

    def getNormedFeatures(self, lstMeanValues):
        """
        returns the features which were normed by dividing them with the mean values

        :param lstMeanValues: mean value of every integer feature
        :return: list of the normed integer features
        """
        lstNormedFeatures = self.getIntegerFeatures()
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

        vectorizer = CountVectorizer(min_df=0.5, vocabulary=lstVocab)

        strFilteredReadme = ""
        strFilteredReadme = self.getFilteredReadme()
        getFilteredRepoDescription = self.getFilteredRepoDescription()
        strFilteredReadme += getFilteredRepoDescription
        strFilteredReadme += getFilteredRepoDescription

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

        iHits = np.sum(self.lstOccurrence)

        # divide each element by a factor to reduce the effectiveness
        iLen = len(strFilteredReadme.split())

        # avoid dividing by 0
        if iLen == 0:
            iLen = 1

        if iHits == 0:
            iHits = 1

        self.dicFoundWords = self.getFeatureOccurences(lstFeatureNames, lstOccurrence, iMinOccurence=1)
        self.printFeatureOccurences(self.dicFoundWords)

        return lstOccurrence

    def getFeatureOccurences(self, lstFeatureNames, lstOccurrence, iMinOccurence=1):
        """
        gets the found words with it's number of occurrences in form of a dictionary

        :param lstFeatureNames: vocab list
        :param lstOccurrence: list of number of occurrences
        :param iMinOccurence: minimum number of hits which are needed
        :return: dictionaryObject
        """
        assert (len(lstFeatureNames) == len(lstOccurrence))
        i = 0
        dicFoundWords = {}
        for iTmpOccurrence in lstOccurrence:
            if iTmpOccurrence > iMinOccurence:
                dicFoundWords[iTmpOccurrence] = lstFeatureNames[i]
            i += 1

        return dicFoundWords

    def getDicFoundWords(self):
        """
        gets the stored dictionary-object
        :return: dictionaryObject
        """

        if self.dicFoundWords is None:
            raise Exception('getWordOccurences() hasn\'t been called yet')
        return self.dicFoundWords

    def printFeatureOccurences(self, dicFoundWords): #lstFeatureNames, lstOccurrence, iMinOccurence=1):
        """
        Prints out every feature with it's number occurence

        :param lstFeatureNames:     list of the given features, these are the column of the sparse-matrix
        :param lstOccurrence:       number of occurence of the individual features (has the same size as lstFeatureName
        :param iMinOccurence:       minimum threshold to print out the feature (if set to 0 all features are print out)
        :return:
        """

        if len(dicFoundWords.items()) > 0:
            strStopper1 = "=" * 80
            strStopper2 = "-" * 80
            print(strStopper2)
            print('detected words from the vocabulary:')

            for k, v in dicFoundWords.items():
                # for more beautiful print layout {:15s} and {:3d} is used
                print('{:15s} {:3f}'.format(v, k))  # {:3d} for integers

            print(strStopper2)

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
