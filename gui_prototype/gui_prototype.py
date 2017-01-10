"""
@file: gui_protoype.py
Created on 07.01.2017 23:06
@project: GitHubRepositoryClassifier

@author: Lukas

GUI Prototype using kivy
"""

from github3.repos.repo import Repository
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
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas     # don't worry, it works even though its red
from kivy.uix.popup import Popup
import clipboard                                                    # pip install clipboard

import sys
import matplotlib.pyplot as plt

import os
from prototype import *
# import prototype.repository_classifier
from prototype.repository_classifier import RepositoryClassifier
from prototype.definitions.categories import CategoryStr
import prototype.github_repo
import webbrowser
from kivy.properties import BooleanProperty

# import prototype.print_overloading

kivy.require("1.9.0")


# http://stackoverflow.com/questions/2297933/creating-a-custom-sys-stdout-class
# other options:
# - redirect_stdout
# - contextlib
# - overload print() function
# ...

class StaticVars:
    b_api_checkbox_state = False                    #checkbox status


class StdOut(object):
    def __init__(self, log_console, oldStdOut):
        # self.txtctrl = txtctrl
        self.log_console = log_console
        self.oldStdOut = oldStdOut

    def write(self, string):
        # self.txtctrl.write(string)
        # try:
        self.log_console.text += string #self.txtctrl.getvalue()
        self.oldStdOut.write(string)
        # except:
        #     pass

    def flush(self):
        pass


class InfoPopup(Popup):
    pass


class SettingsPopup(Popup):
    checkbox_api_token = ObjectProperty()
    token = BooleanProperty(False)

    def __init__(self):
        super(SettingsPopup, self).__init__()
        self.checkbox_api_token.active = StaticVars.b_api_checkbox_state

    def switch_api(self, b_status):
        StaticVars.b_api_checkbox_state = b_status
        print('[INFO] use API updated to: ' + str(b_status))


class FileSaverPopup(Popup):
    filename_input = ObjectProperty()

    log_text = ""

    def save_file(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.log_text)

        self.dismiss()



class GUILayout(BoxLayout):

    # define the ObjectProperties to communicate with the .kv file
    textfield_input = ObjectProperty()                    # user input line
    button_classifier = ObjectProperty()            # the button above the user input. should start the process
    label_info = ObjectProperty()                   # the label underneath the user input. short descriptions go here
    log_console = ObjectProperty()                  # console to output logs. just add text to it.
    label_result = ObjectProperty()                 # the big Result Label in the result corner
    layout_pie_chart = ObjectProperty()             # the Layout for the piechart in the result corner
    layout_diagram1 = ObjectProperty()              # the three TabbedPanelItems to put a diagram, expand if needed
    layout_diagram2 = ObjectProperty()              # ↑
    layout_diagram3 = ObjectProperty()              # ↑

    def show_info(self):
        info_popup = InfoPopup()
        info_popup.open()

    def show_documentation(self):
        webbrowser.open("http://google.com")

    def show_settings(self):
        settings_popup = SettingsPopup()
        settings_popup.open()

    def classify_button_pressed_example(self):                  # THIS IS ONLY THE EXAMPLE! DELETE BEFORE PUBLISHING!
        print("classify-button pressed. Classification started")
        self.button_classifier.text = "pressed!"                        # button demo
        self.button_classifier.disabled = True
        self.label_info.text = "ERROR: Prototype not hooked up yet!"   # error label demo
        print("[ERROR] Prototype not hooked up yet")
        self.label_info.color = 1, 0, 0, 1
        # self.log_console.text = ""                                      # clear console
        self.log_console.scroll_y = 0                             # makes the console scroll down automatically
        # for i in range(0, 50):                                          # demonstrate console
            # self.log_console.text += ("Button pressed, " + str(i) + "\n")

        # ADDING A PIE CHART!
        # The slices will be ordered and plotted counter-clockwise.
        # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        # sizes = [15, 30, 45, 10]

        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs', 'Blob', 'bla', 'blub'
        sizes = [1, 30, 45, 10, 10, 2, 2]

        # colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'black', 'blue', 'red']

        # explode = (0, 0, 0.1, 0)  # only "explode" the 1st slice (i.e. 'Dogs')
        explode = (0, 0, 0.1, 0, 0, 0, 0)  # only "explode" the 1st slice (i.e. 'Dogs')

        # plt.figure(1, figsize=(10, 10), dpi=70)
        plt.figure(1, figsize=(40, 40), dpi=70)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # plt.axis('equal')
        # plt.tight_layout()                                       # http://matplotlib.org/users/tight_layout_guide.html
        fig = plt.gcf()
        fig.patch.set_facecolor('1')
        fig.patch.set_alpha(0.3)

        # plt.show()
        # fig.set_tight_layout(True)

        self.layout_pie_chart.clear_widgets()
        self.layout_pie_chart.add_widget(FigureCanvas(fig))
        # canv.canvas.ask_update()

    def save_log(self):
        save_popup = FileSaverPopup()
        save_popup.log_text = self.log_console.text
        save_popup.open()

    def paste(self):
        # get clipboard data
        self.textfield_input.text = clipboard.paste()
        # print('paste-button pressed')
        # print(clipboard.paste())
        print('pasted text:', clipboard.paste())

    def initialize(self):

        oldStdOut = sys.stdout
        # overload load the sys.strdout to a class-instance of StdOut
        sys.stdout = StdOut(self.log_console, oldStdOut)
        self.log_console.scroll_y = 0                             # makes the console scroll down automatically

        # initialize the repositoryClassifier
        self.repoClassifier = prototype.repository_classifier.RepositoryClassifier(bUseStringFeatures=False, bWithOAuthToken=True)
        self.repoClassifier.loadModelFromFile()

    def validate_url(self, url_in):
        if url_in == "":
            print("[ERROR] Input is empty")
            self.set_error("[ERROR] Input is empty")
            return False
        elif not url_in.startswith("https://"):
            print("[ERROR] Input doesn't start with https://")
            self.set_error("[ERROR] Input doesn't start with https://")
            return False
        elif not url_in.startswith("https://github.com/"):
            print("[ERROR] Input is not a GitHub URL")
            self.set_error("[ERROR] Input is not a GitHub URL")
            return False
        else:
            print("[INFO] Input is a valid URL")
            self.set_info("[INFO] Input is a valid URL")
            return True

    def set_info(self, info):                               # put the info text as info text, color to black
        self.label_info.color = 1, 1, 1, 1
        self.label_info.text = info

    def set_error(self, error):                             # put the info text as error text, color to red
        self.label_info.color = 1, 0, 0, 1
        self.label_info.text = error

    def classify_button_pressed(self):                              # ACTUAL BUTTON CODE
        self.button_classifier.disabled = True                      # disable button

        url_in = "".join(self.textfield_input.text.split())               # read input and remove whitespaces
        print("[INFO] Starting Process with \"" + url_in + "\"")    # print info to console
        valid = self.validate_url(url_in)                           # validate input and handle Errors

        if valid:
            # TODO: Hook up actual code / start classification here
            print("# TODO: start classification here")
            try:
                iLabel, lstFinalPercentages = self.repoClassifier.predictCategoryFromURL(url_in)
                self.renderPlotChar(lstFinalPercentages)
                self.label_result.text = 'Result: ' + CategoryStr.lstStrCategories[iLabel]
                print('iLabel: ', iLabel)
                print('lstFinalPercentages: ', lstFinalPercentages)
            except ArithmeticError:
                print("[ERROR] Repository not found.")
                self.set_error("[ERROR] Repository not found.")
            except Exception as ex:
                print("[ERROR] An unknown Error occurred: " + str(ex))
                self.set_error("[ERROR] An unknown Error occurred")

            self.button_classifier.disabled = False  # re-enable button
        else:
            self.button_classifier.disabled = False                 # re-enable button



    def renderPlotChar(self, lstFinalPercentages):
        # self.log_console.text = ""                                      # clear console
        self.log_console.scroll_y = 0                             # makes the console scroll down automatically
        # for i in range(0, 50):                                          # demonstrate console
            # self.log_console.text += ("Button pressed, " + str(i) + "\n")

        # ADDING A PIE CHART!
        # The slices will be ordered and plotted counter-clockwise.
        # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        labels = CategoryStr.lstStrCategories

        # sizes = [15, 30, 45, 10]

        # multiplicate every element with 100
        lstFinalPercentages[:] = [x * 100 for x in lstFinalPercentages]

        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'gray', 'lightblue', 'tomato']

        # http://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        iMaxIndex = lstFinalPercentages.index(max(lstFinalPercentages))

        lstExplode = [0] * len(lstFinalPercentages)
        lstExplode[iMaxIndex] = 0.1
        explode = lstExplode #(0, 0, 0.1, 0)  # only "explode" the 1st slice (i.e. 'Dogs')
        # self.iFigIndex += 1
        fig = plt.figure(1, figsize=(10, 10), dpi=70)
        fig.clear()
        plt.pie(lstFinalPercentages, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # plt.axis('equal')                                        # this was the actual cause of the resizing !!!
        #  -> this causes a warning; alternative us fig,set_tight_layout(True)
        # plt.tight_layout()                                         # http://matplotlib.org/users/tight_layout_guide.html

        fig = plt.gcf()
        fig.set_tight_layout(True)

        fig.patch.set_facecolor('1')
        fig.patch.set_alpha(0.3)

        # plt.show()

        self.layout_pie_chart.clear_widgets()
        self.layout_pie_chart.add_widget(FigureCanvas(fig))
        # fig.clear()

        # self.layout_pie_chart.add_widget(fig)


class RepositoryClassifierApp(App):
    icon = 'logo_small.png'                          # change window icon
    def build(self):
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        # Window.size = (1200, 800)

        layGUI = GUILayout()
        layGUI.initialize()

        return layGUI


gui = RepositoryClassifierApp()
gui.run()

