import requests
import learning_features
from sklearn.feature_extraction.text import CountVectorizer

import datetime
from os import path
import os
import json

import string_operation
import count_vectorizer_operations

from utility_funcs.io_agent import InputOutputAgent

import numpy as np
from features.learning_features import IntFeatures, StringFeatures


# http://stackoverflow.com/questions/32910096/is-there-a-way-to-auto-generate-a-str-implementation-in-python
def auto_str(cls):
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
        self.user = strUser
        self.name = strName

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

    def getFilteredReadme(self):
        strMyREADME = self.ioAgent.getReadme(self.strDirPath_readme)
        return string_operation.prepare_words(strMyREADME)

    @classmethod
    def fromURL(cls, strURL):
        iIndexUser = 3
        iIndexName = 4
        lststrLabelGroup = strURL.split('/')
        return cls(lststrLabelGroup[iIndexUser], lststrLabelGroup[iIndexName])


    def readAttributes(self):
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
                                       iNumBranches=0,
                                       iSize=self.apiJSON['size'])

        # print(self.apiJSON['contributors_url'])
        # jsContrib = (requests.get(self.apiJSON['contributors_url'])).json()

        # print('len(jsContrib):', len(jsContrib)) # better use subscriber-count ther contributor length only lists the top contributors


    def getFeatures(self):
        lstFeatures = [self.intFeatures.iSubscriberCount,
                       self.intFeatures.iOpenIssues,
                       self.intFeatures.iDevTime,
                       self.intFeatures.dRepoActivity, #dCodeFrequency
                       self.intFeatures.dCommitIntervals,
                       self.intFeatures.iNumBranches,
                       self.intFeatures.iSize
                       ]
        return lstFeatures

    def getNormedFeatures(self, lstMeanValues):
        lstNormedFeatures = self.getFeatures()
        lstNormedFeatures[:] = [x / y for x, y in zip(lstNormedFeatures, lstMeanValues)]
        return lstNormedFeatures

    def getWordOccurences(self, lstVocab):
        vectorizer = CountVectorizer(min_df=0.5, vocabulary=lstVocab)

        strFilteredReadme = self.getFilteredReadme()
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

        # divide each element by a factor to reduce the effectiveness
        iLen = len(strFilteredReadme.split())
        if iLen == 0:
            iLen = 1
        lstOccurrence[:] = [x / iLen * 10 for x in lstOccurrence]

        # count_vectorizer_operations.printFeatureOccurences(lstFeatureNames, lstOccurrence, 2)

        return lstOccurrence


    def getName(self):
        return self.name

    def getUser(self):
        return self.user
