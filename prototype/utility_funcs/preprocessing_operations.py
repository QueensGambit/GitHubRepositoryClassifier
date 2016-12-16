from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

def createVoabularyFeatures(lstRepos):
    strAllReadmes = ""
    lstAllReadmes = []

    for tmpRepo in lstRepos:

        # load the single lines to an array
        #print('tmpRepo.getFilteredREADME(): ', tmpRepo.getFilteredREADME())
        lstAllReadmes.append(tmpRepo.getFilteredReadme())

    # create a counter which counts the occurrence of each word which is defined in the vocabulary
    # by default the vocabulary consists of all words
    # min_df = 3 means that 3 occurrences are needed to appear in the vocab list
    vectorizer = CountVectorizer(min_df=3)

    # return a sparse matrix
    # each column is mapped to a specific feature (see lstFeatureNames)
    # the value describes the occurrence of the word in the current line
    matSparse = vectorizer.fit_transform(lstAllReadmes)

    lstFeatureNames = vectorizer.get_feature_names()

    # Generating word cloud image -> Only 20 plots are supported at once by default


    return lstFeatureNames
