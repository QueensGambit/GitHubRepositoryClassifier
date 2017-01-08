from __future__ import print_function
# Python 3
import builtins as __builtin__

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

import kivy
#kivy.require('1.0.5')

import kivy
from kivy.uix.floatlayout import FloatLayout
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget

from kivy.core.clipboard import Clipboard


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
# fig = plt.figure()
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

# http://stackoverflow.com/questions/101128/how-do-i-read-text-from-the-windows-clipboard-from-python
# from tkinter import Tk
# r = Tk()
# # r.withdraw()
#
# strClipBoardData = r.clipboard_get()
# if strClipBoardData:
#     print('clipboard_data:', strClipBoardData)
# r.clipboard_clear()
# r.destroy()
# r.clipboard_append('i can has clipboardz?')
# print('Clipboard.paste():', Clipboard.paste())


# Alternatives use
# 1) ctypes
#
# import ctypes
#
# CF_TEXT = 1
#
# kernel32 = ctypes.windll.kernel32
# user32 = ctypes.windll.user32
#
# user32.OpenClipboard(0)
# if user32.IsClipboardFormatAvailable(CF_TEXT):
#     data = user32.GetClipboardData(CF_TEXT)
#     data_locked = kernel32.GlobalLock(data)
#     text = ctypes.c_char_p(data_locked)
#     print(text.value)
#     kernel32.GlobalUnlock(data_locked)
# else:
#     print('no text in clipboard')
# user32.CloseClipboard()

# 2) cross-platform Clipboard-Library -> needs to get installed
# import clipboard
#
# clipboard.copy("this text is now in the clipboard")
# strClipBoardText = clipboard.paste()
# if strClipBoardText == '':
#     print('no text in clipboard')
# else:
#     print (clipboard.paste())


# http://stackoverflow.com/questions/550470/overload-print-python
# This must be the first statement before other statements.
# You may only put a quoted or triple quoted string,
# Python comments, other future statements, or blank lines before the __future__ line.




class StaticVars:
    btn1 = None
    strPrintText = ''


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

        StaticVars.btn1 = self.ids.btnTerminal
        StaticVars.btn1.text = 'hello?'
        StaticVars.btn1.bind(on_press=callback)

        self.btnMenu = self.ids.btnMenu
        self.btnMenu.bind(on_press=callback2)


        self.ids.wordCloudBtn.text = '42'

        fl = self.ids.wordCloudPlot
        fl.clear_widgets()
        btn = Button()

        wCanavas = canvas
        # wCanvas2 = canvas2
        wCanavas.size = (3000, 3000)
        # wCanvas2.size = (3000, 3000)
        # wCanavas.size_hint = (0.1, 0.1)
        # fl.add_widget(wCanavas)
        fl.add_widget(FigureCanvas(fig))
        # fl.add_widget(wCanvas2)

        # fl.add_widget(btn)
        fl.add_widget(self.get_fc(1))
        fl.add_widget(self.get_fc(2))
        # fl.add_widget(canvas)

        #self.ids.myBox.orientation = 'horizontal'

        # pass
        print('hello World')


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


def callback2(instance):
    print('sample output')

def callback(instance):
    # print('The button <%s> is being pressed' % instance.text)
    StaticVars.btn1.text = StaticVars.strPrintText

def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    __builtin__.print('My overridden print() function!')

    # if btn1 not None:
    try:
        StaticVars.strPrintText += str(*args) #'test' #*args
        StaticVars.strPrintText += '\n'
        callback(StaticVars.btn1)
        __builtin__.print('update text')
        # StaticVars.btn1 = 'text'
    except:
        pass

    return __builtin__.print(*args, **kwargs)

if __name__ == '__main__':
    #ax = fig.gca()
    myApp().run()
