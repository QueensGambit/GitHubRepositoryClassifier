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

import sys


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    InputOutputAgent.setWithToken(False)
    repoClassifier = RepositoryClassifier(bUseStringFeatures=True)

    # strFilenameCSV = 'example_repos.csv'
    strFilenameCSV = 'additional_data_sets_cleaned.csv'

    #lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData('/data/csv/' + strFilenameCSV)
    #repoClassifier.trainModel(lstTrainData, lstTrainLabels)
    #repoClassifier.exportModelToFile()
    clf = repoClassifier.loadModelFromFile()
    #repoClassifier.predictResultsAndCompare()

    print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
    iLabel, lstFinalPercentages, tmpRepo = repoClassifier.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
    # pobox/overwatch
    # pobox
    #repoClassifier.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
    #repoClassifier.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')

    X = np.empty((10, 3))
    X[0] = np.asarray([tmpRepo.getNumOpenIssue(),
                       tmpRepo.getDevTime(),
                       tmpRepo.getNumWatchers()],
                      dtype=np.float64)
    print(X)

    normalizer = preprocessing.MinMaxScaler()
    normalizer.fit(X)
    X = normalizer.fit_transform(X)

    plot_multi_dim(clf, X)




###############################################################################
# Test
##############################################################################
def plot_multi_dim(clf, multidimarray):
    if not isinstance(multidimarray, (np.ndarray, np.generic)):
        print('Need Numpy')
        return

    if multidimarray.shape[1] > 2:
        plt.cla()
        pca = decomposition.PCA(n_components=2)
        pca.fit(multidimarray)
        multidimarray = pca.transform(multidimarray)

    # plt.scatter(multidimarray[:, 0],
    #             multidimarray[:, 1])
    # plt.show()

    n_digits = 7
    kmeans = KMeans(init='k-means++',n_clusters=n_digits, n_init=10)
    kmeans.fit(multidimarray)

    h = .02

    x_min, x_max = multidimarray[:, 0].min() - 1, multidimarray[:, 0].max() + 1
    y_min, y_max = multidimarray[:, 1].min() - 1, multidimarray[:, 1].max() + 1
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

    plt.scatter(multidimarray[:, 0], multidimarray[:, 1], cmap=plt.cm.Paired)

    centroids = clf.centroids_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=169, linewidths=3,
                color='w', zorder=10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



    # # Plot
    # for i, (clf, y_train) in enumerate((ls50, ls100, rbf_svc, lp100)):
    #     plt.subplot(2, 2, i + 1)
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    #     colors = [color_map[y] for y in y_train]
    #
    #     Z = Z.reshape(xx.shape)
    #     cs = plt.contourf(xx, yy, Z, c=colors, cmap=plt.cm.Paired)
    #     # plt.axis('off')
    #


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