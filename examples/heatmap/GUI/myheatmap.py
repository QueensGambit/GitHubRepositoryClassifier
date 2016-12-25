import matplotlib.pyplot as plt
import pandas

'''
This class creates an heatmap of a soccer game.
'''

class MyHeatMap:

    __iNRows = 600000
    __strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'
    __id = 0
    __data = []

    def __init__(self):
        self.__data = pandas.read_csv(self.__strFile, delimiter=',', nrows=self.__iNRows, header=None, skiprows=500000)

    def set_id(self, id):
        self.__id = id

    def get_figure(self):

        # only get the entries where the id is 4
        dataFilt = self.__data.loc[self.__data[0] == self.__id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]
        z = dataFilt.ix[:, 4]


        # add a description
        #fig = plt.figure(0)
        #fig.canvas.set_window_title('Location_Player_ID_' + str(id))
        plt.hexbin(x, y, bins=None)
        plt.title('Location of Player ID=' + str(self.__id) + '\nwithin the first ' + str(self.__iNRows) + ' Rows (' + str(len(x)) + 'Entries)')
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        plt.colorbar()
        #plt.clf()

        fig = plt.gcf()

        return fig

