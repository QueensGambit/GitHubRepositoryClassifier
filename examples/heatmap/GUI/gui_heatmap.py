# https://kivy.org/docs/guide/lang.html
# integrate matplotlib with kivy
import matplotlib

from xlwt.Bitmap import ObjBmpRecord

matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')


from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.button import Button


from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
import matplotlib.pyplot as plt
import pandas
'''
strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'

iNRows = 600000
id = 38

data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000 ) #,skiprows=500000

# only get the entries where the id is 4
dataFilt = data.loc[data[0] == id]

# .ix returns a the values from  a specific column or row
x = dataFilt.ix[:, 2]
y = dataFilt.ix[:, 3]
z = dataFilt.ix[:, 4]

#other interesting methods:
#x.values, x.toarray, x.transpose, np.random.randn(4242)

#print('x-column only')
#print(x)


# add a description
#fig = plt.figure(0)
#fig.canvas.set_window_title('Location_Player_ID_' + str(id))
plt.hexbin(x, y, bins=None)
plt.title('Location of Player ID=' + str(id) + '\nwithin the first ' + str(iNRows) + ' Rows (' + str(len(x)) + 'Entries)')
plt.xlabel('X-Coordinate')
plt.ylabel('Y-Coordinate')
plt.colorbar()
#plt.clf()

'''

class MyHeatMap:

    __iNRows = 600000
    __strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'
    __id = 0
    __data = []

    def __init__(self):
        self.__data = pandas.read_csv(self.__strFile, delimiter=',', nrows=self.__iNRows, header=None, skiprows=500000)

    def set_id(self, playerid):
        self.__id = playerid

    def clean_plot(self):
        plt.clf()

    def get_figure(self):

        # only get the entries where the id is 4
        dataFilt = self.__data.loc[self.__data[0] == self.__id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]
        z = dataFilt.ix[:, 4]


        # add a description
        fig = plt.figure(0)
        # fig.canvas.set_window_title('Location_Player_ID_' + str(id))
        plt.hexbin(x, y, bins=None)
        plt.title('Location of Player ID=' + str(self.__id) + '\nwithin the first ' + str(self.__iNRows) + ' Rows (' + str(len(x)) + 'Entries)')
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        plt.colorbar()
        #plt.clf()

       # fig = plt.gcf()

        return fig


heat = MyHeatMap()
##default player is player id 4
id = 4
heat.set_id(id)
fig = heat.get_figure()
fig.canvas.set_window_title('Location_Player_ID_' + str(id))
canvas = fig.canvas

class MainWidget(Widget):
    '''Create a controller that receives a custom widget from the kv lang file.

    Add an action to be called from the kv lang file.
    '''

    __player_id = None

    def do_action(self):

        fl = self.ids.wordCloudPlot
        fl.clear_widgets()
        wCanavas = canvas
        fl.add_widget(wCanavas)


    def plotheatmap(self):
        print("plot button pressed")


        if self.__player_id is not None:
            print("playerId: ", self.__player_id)
        else:
            print("no player selected")
        '''
        heat.set_id(self.__player_id)
        fig = heat.get_figure()

        fig.canvas.set_window_title('Location_Player_ID_' + str(id))
        canvas = fig.canvas
        wCanavas = canvas
        '''

    def resetheatmap(self):
        print("reset button pressed")
        heat.clean_plot()


    def getSelectedPlayer(self):
       self.__player_id = self.ids.spinner_id.text

class layout_heatmapApp(App):

    def build(self):
        # wordCloudPlot = ObjectProperty(None)
        myWidget = MainWidget()
        myWidget.do_action()
        return myWidget


    #     boxHeader = BoxLayout(size=root.size, orientation='vertical')
    #
    #
    #     boxMiddle = BoxLayout(size_hint=(1.0, 0.5), orientation='horizontal'
    #     :
    #     boxMiddle.add_widget()
    #
    #     btnMenu = Button(text='menu', size_hint=(0.7, 1.0))
    #
    # btnResult = Button(text='result'
    # size: 100,50
    # size: (root.size / 4),(root.size / 2)
    # pos_hint: { 'center_x' : .5 }


if __name__ == '__main__':
    #ax = fig.gca()
    layout_heatmapApp().run()
