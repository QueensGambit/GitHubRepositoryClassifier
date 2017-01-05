from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import os
import pickle
import logging

def createVoabularyFeatures(lstRepos):
    strAllReadmes = ""
    lstAllReadmes = []

    for tmpRepo in lstRepos:

        # load the single lines to an array
        #print('tmpRepo.getFilteredREADME(): ', tmpRepo.getFilteredREADME())
        lstAllReadmes.append(tmpRepo.getFilteredReadme())

    # create a counter which counts the occurrence of each word which is defined in the vocabulary
    # by default the vocabulary consists of all words
    vectorizer = CountVectorizer(min_df=3)

    # return a sparse matrix
    # each column is mapped to a specific feature (see lstFeatureNames)
    # the value describes the occurrence of the word in the current line
    matSparse = vectorizer.fit_transform(lstAllReadmes)

    lstFeatureNames = vectorizer.get_feature_names()

    # Generating word cloud image -> Only 20 plots are supported at once by default


    return lstFeatureNames


def initInputParameters(strVocabPath, lstGithubRepo):
    """
    Initialies the vocabulary set

    :param strVocabPath:    path were the vocab list is stored
    :param lstGithubRepo:   list of the githubRepository-objects
    :return:
    """

    # generate or read the vocab, depending if the file already exists
    lstVoc = []

    if os.path.isfile(strVocabPath):
        # http://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python
        # read dump file
        with open(strVocabPath, 'rb') as fp:
            logging.debug('open vocab from file...')
            lstVoc = pickle.load(fp)

    else:
        lstVoc = createVoabularyFeatures(lstGithubRepo)
        # dump to file
        with open(strVocabPath, 'wb') as fb:
            logging.debug('dump vocab to file...')
            pickle.dump(lstVoc, fb)

    for tmpGithubRepo in lstGithubRepo:
        lstOccurence = tmpGithubRepo.getWordOccurences(lstVoc)
        # print('sum(sparseMatrix):', sum(matSparse))
        # print('lstOccurence: ', lstOccurence)
        # print('len(Occurence): ', len(lstOccurence))

    return lstVoc
