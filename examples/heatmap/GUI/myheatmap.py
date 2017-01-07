from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas

'''

This class creates an heatmap of soccer data. It was done by the libaries matplotlib, ...
'''
class MyHeatMap:

    __iNRows = 3000000
    __id = 0
    __lstData = None

    def __init__(self, strFile):
        self.__lstData = pandas.read_csv(strFile, delimiter=',', nrows=self.__iNRows, header=None, skiprows=500000)  # ,skiprows=500000

        # Only column 2 and 3 are needed (x, y)
        self.xg = self.__lstData.ix[:, 2]
        self.yg = self.__lstData.ix[:, 3]

        # filter infinite figures otherwise exception
        self.xg = self.xg[np.logical_not(np.isnan(self.xg))]
        self.yg = self.yg[np.logical_not(np.isnan(self.yg))]

        #initialize min and max value of whole data
        self.xgmin = self.xg.min()
        self.xgmax = self.xg.max()
        self.ygmin = self.yg.min()
        self.ygmax = self.yg.max()

    def clean_plot(self):
        plt.clf()

    def get_figure(self, id):

        # only get the entries where the id is 4
        dataFilt = self.__lstData.loc[self.__lstData[0] == id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]
        # z = dataFilt.ix[:, 4]

        # x = y[np.logical_not(np.isnan(x))]
        # y = y[np.logical_not(np.isnan(y))]
        print('lengthX:' + str(len(x)))

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[self.xgmin:self.xgmax:x.size ** 0.5 * 1j, self.ygmin:self.ygmax:y.size ** 0.5 * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        fig = plt.figure(figsize=(9, 10))

        # alpha=0.5 will make the plots semitransparent
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        #set up x,y axes
        plt.xlim(self.xgmin, self.xgmax)
        plt.ylim(self.ygmin, self.ygmax)

        # set axis hidden
        plt.axis('off')

        # overlay soccer field
        im = plt.imread('data\\soccer2.jpg')

        plt.imshow(im, extent=[self.xgmin, self.xgmax, self.ygmin, self.ygmax], aspect='auto')

        fig = plt.gcf()

        return fig