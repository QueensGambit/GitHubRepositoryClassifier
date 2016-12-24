# https://kivy.org/docs/guide/lang.html
# integrate matplotlib with kivy
import matplotlib
from xlwt.Bitmap import ObjBmpRecord

matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import numpy as np
import matplotlib.pyplot as plt
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas

"""
from kivy import BoxLayout

layout = BoxLayout(orientation='vertical')
btn1 = Button(text='Hello')
btn2 = Button(text='World')
layout.add_widget(btn1)
layout.add_widget(btn2)
"""

import kivy
#kivy.require('1.0.5')

import kivy
from kivy.uix.floatlayout import FloatLayout
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget




# The slices will be ordered and plotted counter-clockwise.
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
# fig.set_size_inches(2000, 2000)
# fig.figsi
canvas = fig.canvas


# plt.figure(2)
# plt.subplot(212)
#
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()
#
# fig2 = plt.gcf()
# canvas2 = fig2.canvas

class MainWidget(Widget):
    '''Create a controller that receives a custom widget from the kv lang file.

    Add an action to be called from the kv lang file.
    '''

    def get_fc(self, i):
        fig1 = plt.figure()
        fig1.suptitle('mouse hover over figure or axes to trigger events' +
                      str(i))
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        wid = FigureCanvas(fig1)
        return wid

    def add_plot(self):
        self.add_widget(self.get_fc(1))
        self.add_widget(self.get_fc(2))

    def do_action(self):
        self.ids.wordCloudBtn.text = '42'

        fl = self.ids.wordCloudPlot
        fl.clear_widgets()
        btn = Button()

        wCanavas = canvas
        # wCanvas2 = canvas2
        w = Widget()
        wCanavas.size = (3000, 3000)
        # wCanvas2.size = (3000, 3000)
        # wCanavas.size_hint = (0.1, 0.1)
        fl.add_widget(wCanavas)
        # fl.add_widget(wCanvas2)

        # fl.add_widget(btn)
        fl.add_widget(self.get_fc(1))
        fl.add_widget(self.get_fc(2))
        # fl.add_widget(canvas)

        #self.ids.myBox.orientation = 'horizontal'

        # pass


class myApp(App):

    def build(self):
        # wordCloudPlot = ObjectProperty(None)
        myWidget = MainWidget()
        myWidget.do_action()
        return myWidget #info='Hello world')


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
    myApp().run()
