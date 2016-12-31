import matplotlib.pyplot as plt
import pandas

'''
This class creates an heatmap of a soccer game.
'''

class MyHeatMap:

    __id = 0
    __data = []

    def __init__(self, data):
           self.__data = data

    def clean_plot(self):
        plt.clf()

    def get_figure(self, id):
        self.__id = id

        print(self.__data)
        len(self.__data)

        # only get the entries where the id is 4
        dataFilt = self.__data.loc[self.__data[0] == self.__id]

        # .ix returns a the values from  a specific column or row
        x = dataFilt.ix[:, 2]
        y = dataFilt.ix[:, 3]
        z = dataFilt.ix[:, 4]

        print('lengthX:' + str(len(x)))

        # add a description
        fig = plt.figure(figsize=(9,10))
        fig.canvas.set_window_title('Location_Player_ID_' + str(id))
        plt.hexbin(x, y, bins=None)
        plt.title('Location of Player ID=' + str(id) + '\nwithin the first ' + str(iNRows) + ' Rows (' + str(
            len(x)) + 'Entries)')
        plt.xlabel('X-Coordinate')
        plt.ylabel('Y-Coordinate')
        plt.colorbar()

        # overlay soccer field
        im = plt.imread('soccer.jpg')
        ax1.imshow(im, extent=[0, 740, 0, 515], aspect='auto')
        fig = plt.gcf()

        return fig
