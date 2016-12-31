# https://kivy.org/docs/guide/lang.html
# integrate matplotlib with kivy
import matplotlib

from xlwt.Bitmap import ObjBmpRecord

#set up standard window size
from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
#Config.set('graphics','resizable', False)

matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

import numpy as np

from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.button import Button

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
import matplotlib.pyplot as plt
import pandas


# iNRows = 300000
# strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'
# data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000)
#
# class MyHeatMap:
#
#     __id = 0
#     __data = []
#     size = 0
#     __fig = None
#
#     def __init__(self, data):
#            self.__data = data
#
#     def clean_plot(self):
#         plt.clf()
#
#     def current_state_of_plot(self):
#         self.__fig = plt.gcf()
#         return fig
#
#     def get_figure(self, id):
#         strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'
#
#         iNRows = 300000
#
#         data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000)  # ,skiprows=500000
#
#         print(data)
#         len(data)
#
#         # only get the entries where the id is 4
#         dataFilt = data.loc[data[0] == id]
#
#         # .ix returns a the values from  a specific column or row
#         x = dataFilt.ix[:, 2]
#         y = dataFilt.ix[:, 3]
#         z = dataFilt.ix[:, 4]
#
#         print('lengthX:' + str(len(x)))
#
#         # other interesting methods:
#         # x.values, x.toarray, x.transpose, np.random.randn(4242)
#
#         # print('x-column only')
#         # print(x)
#
#
#         # hist2d is a non hex-represantation
#         # plt.hist2d(x, y, bins=50)
#
#         # add a description
#         self.__fig = plt.figure(0)
#         self.__fig.canvas.set_window_title('Location_Player_ID_' + str(id))
#         plt.hexbin(x, y, bins=None)
#         plt.title('Location of Player ID=' + str(id) + '\nwithin the first ' + str(iNRows) + ' Rows (' + str(
#             len(x)) + 'Entries)')
#         plt.xlabel('X-Coordinate')
#         plt.ylabel('Y-Coordinate')
#         plt.colorbar()
#         self.__fig = plt.gcf()
#
#         return self.__fig
#
# heat = MyHeatMap(data)
#
# #get plot with player_id 4
# fig = heat.get_figure(4)


strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'

iNRows = 10000
id = 4

data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000 ) #,skiprows=500000

print(data)
len(data)

# only get the entries where the id is 4
dataFilt = data.loc[data[0] == id]


# .ix returns a the values from  a specific column or row
x = dataFilt.ix[:, 2]
y = dataFilt.ix[:, 3]
#z = dataFilt.ix[:, 4]

print('lengthX:' + str(len(x)))

#other interesting methods:
#x.values, x.toarray, x.transpose, np.random.randn(4242)

#print('x-column only')
#print(x)


# hist2d is a non hex-represantation
#plt.hist2d(x, y, bins=50)
fig = plt.figure(figsize=(9, 10))
fig.canvas.set_window_title('Location_Player_ID_' + str(id))
plt.hexbin(x, y, bins=None)
plt.title('Location of Player ID=' + str(id) + '\nwithin the first ' + str(iNRows) + ' Rows (' + str(
len(x)) + 'Entries)')
plt.xlabel('X-Coordinate')
plt.ylabel('Y-Coordinate')
plt.colorbar()
z = np.exp(-((x-1)**2+y**2))

# Plot the density map using nearest-neighbor interpolation
plt.pcolormesh(x,y,z)
plt.colorbar()
# overlay soccer field
#im = plt.imread('data\\soccer.jpg')
#ax1 = fig.add_subplot(211)
#ax1.imshow(im, aspect='auto')



canvas = fig.canvas


class MainWidget(Widget):
    '''Create a controller that receives a custom widget from the kv lang file.

    Add an action to be called from the kv lang file.
    '''

    __player_id = None
    fl = None
    wCanvas = None

    def do_action(self):

        self.fl = self.ids.wordCloudPlot
        self.fl.clear_widgets()
        self.wCanavas = canvas
        self.wCanavas.canvas.ask_update()
        self.fl.add_widget(self.wCanavas)

    def plotheatmap(self):
        print("plot button pressed")


        if self.__player_id is not None:

            # heat.clean_plot()
            # fig = heat.get_figure(self.__player_id)
            # canvas = fig.canvas
            #
            # self.fl = self.ids.wordCloudPlot
            # self.fl.clear_widgets()

            # The slices will be ordered and plotted counter-clockwise.
           # heat.clean_plot()
            self.fl.clear_widgets()
            labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
            sizes = [15, 30, 45, 10]
            colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
            explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

            plt.figure(1)
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            # Set aspect ratio to be equal so that pie is drawn as a circle.
            plt.axis('equal')

            # fig = plt.figure()
            fig = plt.gcf()
            plt.savefig('debugplot.png')
            # fig.set_size_inches(2000, 2000)
            # fig.figsi
            canvas = fig.canvas

            self.wCanavas = canvas
            self.wCanavas.canvas.ask_update()

            self.fl.add_widget(self.wCanavas)

        else:
            print("no player selected")


    def resetheatmap(self):
        print("reset button pressed")
        heat.clean_plot()
        self.wCanavas.canvas.ask_update()



    def getSelectedPlayer(self):
       self.__player_id = self.ids.spinner_id.text

class layout_heatmapApp(App):

    def build(self):
        # wordCloudPlot = ObjectProperty(None)
        myWidget = MainWidget()
        myWidget.do_action()
        return myWidget


if __name__ == '__main__':
    #ax = fig.gca()
   layout_heatmapApp().run()
