"""
@file: main.py.py
Created on 07.01.2017 18:20
@project: GitHubRepositoryClassifier

@author: Anonym

Sample usage of the repository-classifier
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans

from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent
import matplotlib.patches as mpatches

import sys
from prototype.definitions.categories import CategoryStr


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    InputOutputAgent.setWithToken(True)
    repoClassifier = RepositoryClassifier(bUseStringFeatures=True)

    # strFilenameCSV = 'example_repos.csv'
    strFilenameCSV = 'additional_data_sets_cleaned.csv'

    lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData('/data/csv/' + strFilenameCSV)
    repoClassifier.trainModel(lstTrainData, lstTrainLabels)
    repoClassifier.exportModelToFile()
    clf, lstMeanValues, matIntegerTrainingData, lstTrainLabels, lstTrainData, normalizer = repoClassifier.loadModelFromFile()
    repoClassifier.predictResultsAndCompare()

    print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
    iLabel, iLabelAlt, lstFinalPercentages, tmpRepo = repoClassifier.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
    # pobox/overwatch
    # pobox
    #repoClassifier.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
    #repoClassifier.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')

    print(matIntegerTrainingData)
    plot_multi_dim(clf, lstTrainData, lstTrainLabels)



def plot_multi_dim(clf, data, lstTrainLabels):

    # normalizer = preprocessing.MinMaxScaler()
    # normalizer = preprocessing.RobustScaler()
    # normalizer = preprocessing.StandardScaler()
    normalizer = preprocessing.Normalizer()

    normalizer.fit(data)
    data = normalizer.fit_transform(data)

    if not isinstance(data, (np.ndarray, np.generic)):
        print('Need Numpy')
        return

    if len(data) < 2:
        print('Need more values')
        return

    if data.shape[1] > 2:
        plt.cla()
        pca = decomposition.PCA(n_components=2)
        pca.fit(data)
        data = pca.transform(data)

    n_clusters = 7
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(data)
    h = .02

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    #plt.plot(multidimarray[:, 0], multidimarray[:, 1], 'k.', markersize=2)

    lstColors = [None] * len(lstTrainLabels)
    lstStrLabels = [None] * len(lstTrainLabels)

    for i, iLabel in enumerate(lstTrainLabels):
        lstColors[i] = CategoryStr.lstStrColors[iLabel]
        lstStrLabels = CategoryStr.lstStrCategories[iLabel]

    plt.scatter(data[:, 0], data[:, 1], cmap=plt.cm.Paired, color=lstColors)

    centroids = clf.centroids_
    centroids = normalizer.fit_transform(centroids)


    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color=CategoryStr.lstStrColors, zorder=10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    lstPatches = [None] *len(CategoryStr.lstStrCategories)
    for i, strCategory in enumerate(CategoryStr.lstStrCategories):
        lstPatches[i] = mpatches.Patch(color=CategoryStr.lstStrColors[i], label=strCategory)

    plt.legend(handles=lstPatches)

    plt.show()



#### RESULTS
# fPredictionRes: 0.612903225806
# fAccuracy:  61.2903225806 %
# NearestCentroid()
# fPredictionRes: 0.58064516129
# fAccuracy:  58.064516129 %
#
#
# KNeighborsClassifier()
# fPredictionRes: 0.41935483871
# fAccuracy:  41.935483871 %
#
#
# RadiusNeighborsClassifier()
#
#
# -- > without removing stop words and with length > 3
# fPredictionRes: 0.645161290323
# fAccuracy:  64.5161290323 %
# --> this is the best result but it doesn't feel right,
#  due to it's randomness of stopper words
#
# ---> lenght < 2 and with removing stop words
# fPredictionRes: 0.612903225806
# fAccuracy:  61.2903225806 %


if __name__ == "__main__":
    main()