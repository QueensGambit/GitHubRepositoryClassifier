# https://kivy.org/docs/guide/lang.html
# integrate matplotlib with kivy
import matplotlib

from xlwt.Bitmap import ObjBmpRecord

#set up standard window size
from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '600')
Config.set('graphics','resizable', False)
#matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
import pandas
import timeit

class MyHeatMap:
    __iNRows =4700000
    __lstData = None

    def __init__(self, strFile):
        start = timeit.timeit()

        self.__lstData = pandas.read_csv(strFile, delimiter=',', nrows=self.__iNRows, header=None,
                                         skiprows=500000)  # ,skiprows=500000

        end = timeit.timeit()
        time = end - start
        print("time read_csv: ", time)
        # get a dupletfree list of player id's to set up gui elements
        self.__lstplayer = list(set(self.__lstData.ix[:, 0]))

        #columns x and y
        self.__xg = self.__lstData.ix[:, 2]
        self.__yg = self.__lstData.ix[:, 3]

        # filter infinite figures otherwise exception
        self.__xg = self.__xg[np.logical_not(np.isnan(self.__xg))]
        self.__yg = self.__yg[np.logical_not(np.isnan(self.__yg))]

        start = timeit.timeit()

        # initialize min and max value of whole data
        self.__xgmin = self.__xg.min()
        self.__xgmax = self.__xg.max()
        self.__ygmin = self.__yg.min()
        self.__ygmax = self.__yg.max()

        end = timeit.timeit()
        time = end - start
        print("min and max values: ", time)

    def clean_plot(self):
        plt.clf()


    def get_playerlist(self):
        return self.__lstplayer

    def get_figure(self, pid):
        start = timeit.timeit()
        #id = pid
        id = int(pid)
        # only get the entries where the id is 4
        dataFilt = self.__lstData.loc[self.__lstData[0] == id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]


        # x = y[np.logical_not(np.isnan(x))]
        # y = y[np.logical_not(np.isnan(y))]


        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[self.__xgmin:self.__xgmax:x.size ** 0.5 * 1j, self.__ygmin:self.__ygmax:y.size ** 0.5 * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        fig = plt.figure(figsize=(9, 10))

        # alpha=0.5 will make the plots semitransparent
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        # set up x,y axes
        plt.xlim(self.__xgmin, self.__xgmax)
        plt.ylim(self.__ygmin, self.__ygmax)

        # set axis hidden
        #plt.axis('off')

        # turn off interactive plotting - speeds things up by 1 Frame / second
        plt.ioff()

        # overlay soccer field
        im = plt.imread('data\\soccer2.jpg')
        plt.imshow(im, extent=[self.__xgmin, self.__xgmax, self.__ygmin, self.__ygmax], aspect='auto')

        fig = plt.gcf()
        # We need to draw the canvas before we start animating...
        end = timeit.timeit()
        time = end - start
        print("creating plot ", time)


        return fig


strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'
heat = MyHeatMap(strFile)
fig = heat.get_figure(38)
canvas = fig.canvas

class MainWidget(Widget):
    '''Create a controller that receives a custom widget from the kv lang file.

    Add an action to be called from the kv lang file.
    '''

    __player_id = None
    fl = None
    wCanvas = None

    def do_action(self):


        spinner = self.ids.playerlist
        strPlayerlist = map(str, heat.get_playerlist())
        spinner.values = strPlayerlist

        self.fl = self.ids.wordCloudPlot
        self.fl.clear_widgets()
        self.wCanavas = canvas
        self.wCanavas.canvas.ask_update()
        self.fl.add_widget(self.wCanavas)

    def plotheatmap(self):
        print("plot button pressed")

        if self.__player_id is not None:

            self.fl.clear_widgets()
            heat.clean_plot()
            fig = heat.get_figure(self.__player_id)

            canvas = fig.canvas
            self.wCanavas = canvas
            self.wCanavas.canvas.ask_update()
            self.fl.add_widget(self.wCanavas)

        else:
            print("no player selected")


    def resetheatmap(self):
        print("reset button pressed")
        heat.clean_plot()
        self.fl.clear_widgets()
        self.wCanavas.canvas.ask_update()

    def getSelectedPlayer(self):
       self.__player_id = self.ids.playerlist.text

class layout_heatmapApp(App):

    def build(self):
        # wordCloudPlot = ObjectProperty(None)
        myWidget = MainWidget()
        myWidget.do_action()
        return myWidget


if __name__ == '__main__':
   layout_heatmapApp().run()
