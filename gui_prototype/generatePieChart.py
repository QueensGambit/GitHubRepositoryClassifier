# integrate matplotlib with kivy
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import numpy as np
import matplotlib.pyplot as plt

class PieChart:


    def __init__(self):
        pass


    def getFigure(self):
        # The slices will be ordered and plotted counter-clockwise.
        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        sizes = [15, 30, 45, 10]
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')

        # fig = plt.figure()
        fig = plt.gcf()

        return fig