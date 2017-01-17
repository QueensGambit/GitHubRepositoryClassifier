"""
@file: main.py.py
Created on 07.01.2017 18:20
@project: GitHubRepositoryClassifier

@author: Anonym

Sample usage of the repository-classifier
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.semi_supervised import label_propagation
from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent
from prototype.github_repo import GithubRepo
from matplotlib.colors import colorConverter

from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
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

    #lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData('/data/csv/' + strFilenameCSV)
    #repoClassifier.trainModel(lstTrainData, lstTrainLabels)
    #repoClassifier.exportModelToFile()
    clf, lstMeanValues, matIntegerTrainingData, lstTrainLabels, lstTrainData, normalizer, normalizerIntegerAttr, lstTrainDataRaw = repoClassifier.loadModelFromFile()
    #repoClassifier.predictResultsAndCompare()

    print('Raw: ', lstTrainDataRaw)
    print('~~~~~~~~~~~~~ PREDICTION FROM SINGLE URL ~~~~~~~~~~~~~~~')
    iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL('https://github.com/akitaonrails/vimfiles')
    # pobox/overwatch
    # pobox
    #repoClassifier.predictCategoryFromOwnerRepoName('pobox', 'overwatch')
    #repoClassifier.predictCategoryFromOwnerRepoName('QueensGambit', 'Barcode-App')

    print('len(lstTrainData): ', len(lstTrainData))
    print('len(lstTrainData[0): ', len(lstTrainData[0]))

    print('lstTrainData:', lstTrainData)
    # matIntegerTrainingData = normalizer.transform(matIntegerTrainingData)

    #plot_multi_dim(clf, lstTrainData, lstTrainLabels)
    # semisupervised(matIntegerTrainingData)
    semisupervised(lstTrainData)


def plot_multi_dim(clf, data, lstTrainLabels):

    # normalizer = preprocessing.MinMaxScaler()
    # normalizer = preprocessing.RobustScaler()
    # normalizer = preprocessing.StandardScaler()
    # normalizer = preprocessing.Normalizer()
    #
    # normalizer.fit(data)
    # data = normalizer.fit_transform(data)

    if data.shape[1] > 2:
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

#    for i, iLabel in enumerate(lstTrainLabels):
#        lstColors[i] = CategoryStr.lstStrColors[iLabel]
#        lstStrLabels = CategoryStr.lstStrCategories[iLabel]

    # plt.scatter(data[:, 0], data[:, 1], cmap=plt.cm.Paired, color=lstColors)
    plt.scatter(data[:, 0], data[:, 1], cmap=plt.cm.Paired)

    # if clf is not None:
    #     centroids = clf.centroids_
    #     centroids = normalizer.fit_transform(centroids)
    #
    #     plt.scatter(centroids[:, 0], centroids[:, 1],
    #                 marker='x', s=169, linewidths=3,
    #                 color=CategoryStr.lstStrColors, zorder=10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    lstPatches = [None] *len(CategoryStr.lstStrCategories)
    for i, strCategory in enumerate(CategoryStr.lstStrCategories):
        lstPatches[i] = mpatches.Patch(color=CategoryStr.lstStrColors[i], label=strCategory)

    plt.legend(handles=lstPatches)

    plt.show()


def semisupervised(matIntegerTrainingData):

    strProjectDir = str(Path().resolve().parent)
    strProjPathFileNameCSV = '/data/csv/additional_data_sets_cleaned.csv'
    trainData = pd.read_csv(strProjectDir + strProjPathFileNameCSV, header=0, delimiter=",")
    lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']
    lstGithubRepo = []

    length = len(matIntegerTrainingData)
    iNumTrainData = len(trainData.index)
    X = matIntegerTrainingData
    y = np.empty(length)

    for i in range(iNumTrainData):
        # lstGithubRepo.append(GithubRepo.fromURL(trainData["URL"][i]))  # skip this for now for a faster run time
        pass

    for i in range(length):
        if i % 2 == 0:
            value = lstStrCategories.index(trainData["CATEGORY"][i])
        else:
            value = -1

        y[i] = np.asarray([value], dtype=np.int)

    plot(X, y, lstStrCategories)


def plot(X, y, lstStrCategories):

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    rng = np.random.RandomState(0)

    y_30 = np.copy(y)
    y_30[rng.rand(len(y)) < 0.3] = -1
    y_50 = np.copy(y)
    y_50[rng.rand(len(y)) < 0.5] = -1
    y_75 = np.copy(y)
    y_75[rng.rand(len(y)) < 0.8] = -1

    ls50 = (label_propagation.LabelSpreading().fit(X, y_50), y_50)
    ls75 = (label_propagation.LabelSpreading().fit(X, y_75), y_75)
    ls100 = (label_propagation.LabelSpreading().fit(X, y), y)
    lp100 = (label_propagation.LabelPropagation().fit(X, y), y)

    clfLabelSpread = label_propagation.LabelSpreading()
    clfLabelSpread.fit(X, y_30)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))

    titles = ['Label Spreading 50%',
              'Label Spreading 75%',
              'Label Spreading 100%',
              'Label Propagation 100%']

    color_map = {-1: (1, 1, 1),
                 0: colorConverter.to_rgb(CategoryStr.lstStrColors[0]),
                 1: colorConverter.to_rgb(CategoryStr.lstStrColors[1]),
                 2: colorConverter.to_rgb(CategoryStr.lstStrColors[2]),
                 3: colorConverter.to_rgb(CategoryStr.lstStrColors[3]),
                 4: colorConverter.to_rgb(CategoryStr.lstStrColors[4]),
                 5: colorConverter.to_rgb(CategoryStr.lstStrColors[5]),
                 6: colorConverter.to_rgb(CategoryStr.lstStrColors[6])}

    cs = None

    for i, (clf, y_train) in enumerate((ls50, ls75, ls100, lp100)):
        plt.subplot(2, 2, i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        colors = [color_map[y] for y in y_train]

        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, c=CategoryStr.lstStrColors, cmap=plt.cm.Paired)
        #plt.axis('off')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

        plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=plt.cm.Paired, s=80)

        plt.title(titles[i])

    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
        for pc in cs.collections]

    matPredictRes = clfLabelSpread.predict(X)
    print('matPredictRes: ', matPredictRes)

    plt.legend(proxy, lstStrCategories)
    plt.show()



#### RESULTS


if __name__ == "__main__":
    main()