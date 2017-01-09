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
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from myheatmap import MyHeatMap


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
