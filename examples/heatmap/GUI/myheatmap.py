from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas

'''

This class creates an heatmap of soccer data. It was done by the libaries matplotlib,
'''

class MyHeatMap:

    __iNRows =3000000
    __lstData = None

    def __init__(self, strFile):

        self.__lstData = pandas.read_csv(strFile, delimiter=',', nrows=self.__iNRows, header=None,
                                         skiprows=500000)

        # get a dupletfree list of player id's to set up gui elements
        self.__lstplayer = list(set(self.__lstData.ix[:, 0]))

        #coordinates are from: http://debs.org/?p=41
        self.__xgmin = -52489
        self.__xgmax = 52489
        self.__ygmin = -33939
        self.__ygmax = 33941

    def clean_plot(self):
        plt.clf()


    def get_playerlist(self):
        return self.__lstplayer

    def get_figure(self, pid):

        #id = pid
        id = pid
        # only get the entries where the id is 4
        dataFilt = self.__lstData.loc[self.__lstData[0] == id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]


        # x = y[np.logical_not(np.isnan(x))]
        # y = y[np.logical_not(np.isnan(y))]

        #Kernel Density Estimation via Scipy Lib
        #generell: each point is a rectangle
        #if rectangles overlape so their colors getting added.
        #-->Smooth picture
        #-->no end points
        #-->depend on bandwidth

        k = gaussian_kde(np.vstack([x,  y]))
        xi, yi = np.mgrid[self.__xgmin:self.__xgmax:x.size ** 0.5 * 1j, self.__ygmin:self.__ygmax:y.size ** 0.5 * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Alternative
        #-->not smooth pictures
        #-->depend on end points of bins
        #-->depend on width of bins
        # plt.hist2d([x,y)

        fig = plt.figure(figsize=(9, 10))
        # alpha=0.5 will make the plots semitransparent
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        # set up x,y axes
        plt.xlim(self.__xgmin, self.__xgmax)
        plt.ylim(self.__ygmin, self.__ygmax)

        # set axis hidden
        plt.axis('off')

        # turn off interactive plotting - speeds things up by 1 Frame / second
        plt.ioff()

        # overlay soccer field
        im = plt.imread('data\\soccer2.jpg')
        plt.imshow(im, extent=[self.__xgmin, self.__xgmax, self.__ygmin, self.__ygmax], aspect='auto')

        fig = plt.gcf()
        # We need to draw the canvas before we start animating...



        return fig
