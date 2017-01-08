"""
@file: gui_protoype.py
Created on 07.01.2017 23:06
@project: GitHubRepositoryClassifier

@author: Lukas

GUI Prototype using kivy
"""

from __future__ import print_function
# used imports to overload the print function in Python 3.X
import builtins as __builtin__

from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
# Config.set('graphics', 'resizable', False)


import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from kivy.uix.popup import Popup
import clipboard                                                    # pip install clipboard


import matplotlib.pyplot as plt

import os

kivy.require("1.9.0")


class StaticVars:
    # the tempory print-output of s single print() call is stored in a static string variable
    strPrintText = ''
    # this is satic the widget for the console
    widgetConsole = None

class FileSaverPopup(Popup):
    filename_input = ObjectProperty()

    log_text = ""

    def save_file(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.log_text)

        self.dismiss()

#http://stackoverflow.com/questions/550470/overload-print-python
# overload the print() function to log to the console
def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    # __builtin__.print('My overridden print() function!')

    try:
        # concat all arguments together
        for i, arg in enumerate(args):
            # add a space in between the arguments
            if i != 0:
                StaticVars.strPrintText += " "
            # add the argument to the string
            StaticVars.strPrintText += str(arg)

        # add a new line after the print statement
        StaticVars.strPrintText += '\n'
        updateConsole()
        __builtin__.print('update text')
    except:
        pass

    return __builtin__.print(*args, **kwargs)


def updateConsole():
    StaticVars.widgetConsole.text += StaticVars.strPrintText

class GUILayout(BoxLayout):

    # define the ObjectProperties to communicate with the .kv file
    url_input = ObjectProperty()                    # user input line
    classifier_button = ObjectProperty()            # the button above the user input. should start the process
    error_label = ObjectProperty()                  # the label underneath the user input. short descriptions go here
    # log_console = ObjectProperty()                  # console to output logs. just add text to it.
    result_label = ObjectProperty()                 # the big Result Label in the result corner
    pie_chart = ObjectProperty()                    # the Layout for the piechart in the result corner
    diagram1 = ObjectProperty()                     # the three TabbedPanelItems to put a diagram, expand if needed
    diagram2 = ObjectProperty()                     # ↑
    diagram3 = ObjectProperty()                     # ↑
    dialabel1 = ObjectProperty()

    def classify_button_pressed(self):
        print("classify-button pressed!")
        self.classifier_button.text = "pressed!"                        # button demo
        self.classifier_button.disabled = True
        self.error_label.text = "ERROR: Prototype not hooked up yet!"   # error label demo
        self.error_label.color = 1, 0, 0, 1
        # self.log_console.text = ""                                      # clear console
        StaticVars.widgetConsole.scroll_y = 0                                   # makes the console scroll down automatically
        # for i in range(0, 50):                                          # demonstrate console
            # self.log_console.text += ("Button pressed, " + str(i) + "\n")

        button = Button(text='Testbutton')                              # demonstration of clearing an area and adding a
        self.pie_chart.clear_widgets()                                  # widget to it
        self.pie_chart.add_widget(button)

        # ADDING A PIE CHART!
        # The slices will be ordered and plotted counter-clockwise.
        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        sizes = [15, 30, 45, 10]
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        explode = (0, 0, 0.1, 0)  # only "explode" the 1st slice (i.e. 'Dogs')
        plt.figure(1, figsize=(10, 10), dpi=70)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.tight_layout()                                         # http://matplotlib.org/users/tight_layout_guide.html
        fig = plt.gcf()
        fig.patch.set_facecolor('1')
        fig.patch.set_alpha(0.3)

        # plt.show()

        self.pie_chart.clear_widgets()
        self.pie_chart.add_widget(FigureCanvas(fig))
        # canv.canvas.ask_update()

    def save_log(self):
        save_popup = FileSaverPopup()
        save_popup.log_text = self.log_console.text
        save_popup.open()

    def paste(self):
        # get clipboard data
        self.url_input.text = clipboard.paste()
        # print('paste-button pressed')
        # print(clipboard.paste())
        print('pasted text:', clipboard.paste())

    def initialize(self):
        StaticVars.widgetConsole = self.ids.log_console  # connect the console object with the static variable

class GUIApp(App):
    def build(self):
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        # Window.size = (1200, 800)

        layGUI = GUILayout()
        layGUI.initialize()

        return layGUI

gui = GUIApp()
gui.run()
