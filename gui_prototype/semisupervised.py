import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from matplotlib.pyplot import legend
import matplotlib.patches as mpatches
import math
import os
from sklearn.semi_supervised import label_propagation
from prototype.repository_classifier import RepositoryClassifier
from prototype.github_repo import GithubRepo
import gui_prototype



strProjectDir = str(Path().resolve().parent)
dir = str(Path(__file__).parents[0])
file1 = '\semisupervised_ghtorrent.csv'
file2 = '\semisupervised_ghtorrent2.csv'
strProjPathFileNameCSV ='/data/csv/additional_data_sets_cleaned.csv'

trainData = pd.read_csv(strProjectDir + strProjPathFileNameCSV, header=0, delimiter=",")
iNumTrainData = len(trainData.index)

data1 = pd.read_csv(dir + file1, header=0, delimiter=",")
data2 = pd.read_csv(dir + file2, header=0, delimiter=",")

lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']
lstGithubRepo = []

for i in range(iNumTrainData):
    lstGithubRepo.append(GithubRepo.fromURL(trainData["URL"][i]))


length = len(data1) + len(lstGithubRepo)
i_Offset = len(data1)
dimension = 3

X = np.empty((length, dimension))
y = np.empty(length)

for i in range(len(data1)):

    X[i] = np.asarray([data1["`num_issues`"][i],
                       data1["`dev_time_days`"][i],
                       data2["`num_watchers`"][i]],
                      dtype=np.float64)
    y[i] = np.asarray([-1],
                       dtype=np.int)

for i, tmpRepo in enumerate(lstGithubRepo):
    X[i + i_Offset] = np.asarray([tmpRepo.getNumOpenIssue(),
                                  tmpRepo.getDevTime(),
                                  tmpRepo.getNumWatchers()],
                                 dtype=np.float64)
    y[i + i_Offset] = np.asarray([lstStrCategories.index(trainData["CATEGORY"][i])],
                        dtype=np.int)

# normalizer = preprocessing.Normalizer()
normalizer = preprocessing.MinMaxScaler()
# normalizer = preprocessing.StandardScaler()
normalizer.fit(X)
X = normalizer.fit_transform(X)
# X = preprocessing.normalize(X)

# clf = label_propagation.LabelPropagation()
# clf.fit(X, y)
# print(clf.predict(X[:500]))

# clf = label_propagation.LabelSpreading()
# clf.fit(X, y)
# print(clf.predict(X))

print(X)
plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
print(X)


X = np.empty((10, 2))
X[0] = np.asarray([0., 0.], dtype=np.float64)
X[1] = np.asarray([0., 1.], dtype=np.float64)
X[2] = np.asarray([0., 2.], dtype=np.float64)
X[3] = np.asarray([1., 0.], dtype=np.float64)
X[4] = np.asarray([1., 1.], dtype=np.float64)
X[5] = np.asarray([1., 2.], dtype=np.float64)
X[6] = np.asarray([2., 0.], dtype=np.float64)
X[7] = np.asarray([2., 1.], dtype=np.float64)
X[8] = np.asarray([2., 2.], dtype=np.float64)
X[9] = np.asarray([3, 3], dtype=np.float64)

print(X)
normalizer.fit(X)
X = normalizer.fit_transform(X)
print(X)


rng = np.random.RandomState(0)

y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1

ls50 = (label_propagation.LabelSpreading().fit(X, y_50), y_50)
ls100 = (label_propagation.LabelSpreading().fit(X, y), y)
lp100 = (label_propagation.LabelPropagation().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf').fit(X, y), y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))

titles = ['Label Spreading 50%',
          'Label Spreading 100%',
          'SVC with rbf kernel',
          'Label Propagation 100%']


color_map = {-1: (1, 1, 1),
             0: (0, 0, .9),
             1: (1, 0, 0),
             2: (.8, .6, 0),
             3: (0, 1, 0),
             4: (1, .9, .5),
             5: (.5, .5, .5),
             6: (.7, .7, .7)}

cs = None

# Plot
for i, (clf, y_train) in enumerate((ls50, ls100, rbf_svc, lp100)):
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    colors = [color_map[y] for y in y_train]

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, c=colors, cmap=plt.cm.Paired)
    # plt.axis('off')

    plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=plt.cm.Paired)

    plt.title(titles[i])

# legend for contours
# http://stackoverflow.com/questions/10490302/how-do-you-create-a-legend-for-a-contour-plot-in-matplotlib
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]

plt.legend(proxy, lstStrCategories)

# Plot training points
# Unlabeled points are colored white
plt.show()






