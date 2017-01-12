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

import sys, os
import matplotlib.pyplot as plt

# add the current directory to the system path in order to find the modules in relative path
# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.abspath("../prototype"))

# import prototype.print_overloading

# from prototype import *
# import prototype.repository_classifier
from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent       # this import is used to enable or disable the GithubToken

from prototype.definitions.categories import CategoryStr
# import prototype.github_repo
import webbrowser
from kivy.properties import BooleanProperty

# threading and animation
# multithreading in kivy:
# https://github.com/kivy/kivy/wiki/Working-with-Python-threads-inside-a-Kivy-application
import time
import threading
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.factory import Factory

from wordcloud import WordCloud
from prototype.definitions.categories import Category

from PIL import Image
from PIL import ImageOps
import numpy as np

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
    label_api_error = ObjectProperty()

    def __init__(self):
        super(SettingsPopup, self).__init__()
        self.checkbox_api_token.active = StaticVars.b_api_checkbox_state

    def switch_api(self, b_status):     # TODO: This will pass the test the second time. now the app gets stuck
        try:
            self.label_api_error.text = ""
            InputOutputAgent.setWithToken(b_status)
            StaticVars.b_api_checkbox_state = b_status
            print('[INFO] Use API updated to: ' + str(b_status))
        except ConnectionError as ce:
            self.label_api_error.text = "[ERROR] No Connection could be established."
            print("[ERROR] No Connection could be established: " + str(ce))
            self.checkbox_api_token.active = False
            StaticVars.b_api_checkbox_state = False


class FileSaverPopup(Popup):
    filename_input = ObjectProperty()
    label_save = ObjectProperty()

    log_text = ""

    def save_file(self, path, filename):
        try:
            with open(os.path.join(path, filename), 'w') as stream:
                stream.write(self.log_text)
            self.dismiss()
            print("[INFO] Logfile saved to: " + path + "\\" + filename)
        except PermissionError as err:
            print("[ERROR] Logfile couldn't be saved. Permission denied. Path: " + path + "\nError: " + str(err))
            self.label_save.text = "[ERROR] Couldn't save. Permission denied."


class GUILayout(BoxLayout):

    stop = threading.Event()

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

    # threading
    def start_classification_thread(self, l_text, url_in):
        threading.Thread(target=self.classification_thread, args=(l_text, url_in)).start()

    def classification_thread(self, l_text, url_in):
        # Remove a widget, update a widget property, create a new widget,
        # add it and animate it in the main thread by scheduling a function
        # call with Clock.
        Clock.schedule_once(self.start_test, 0)

        iLabel = None
        lstFinalPercentages = []
        try:
            iLabel, lstFinalPercentages, tmpRepo = self.repoClassifier.predictCategoryFromURL(url_in)
        except ConnectionError:
            print("[ERROR] A connection error occurred")
            self.set_error("[ERROR] A connection error occurred")
        except Exception as ex:
            print("[ERROR] An unknown Error occurred: " + str(ex))
            self.set_error("[ERROR] An unknown Error occurred")

        # print('iLabel:', iLabel)
        # Do some thread blocking operations.
        # time.sleep(5)
        # l_text = str(int(label_text) * 3000)

        # Update a widget property in the main thread by decorating the
        # called function with @mainthread.
        # self.update_label_text(l_text)

        # Do some more blocking operations.
        # time.sleep(2)

        # Remove some widgets and update some properties in the main thread
        # by decorating the called function with @mainthread.
        self.show_classification_result(iLabel, lstFinalPercentages)

        # Start a new thread with an infinite loop and stop the current one.
        # threading.Thread(target=self.infinite_loop).start()

        strText = str(tmpRepo.getFilteredReadme(bApplyStemmer=True) + " " + tmpRepo.getFilteredRepoDescription(bApplyStemmer=True))
        self.show_wordcloud(strText, iLabel)

    def start_test(self, *args):
        self.button_classifier.disabled = True                      # disable button

        # Remove the button.
        self.layout_pie_chart.clear_widgets()
        # self.remove_widget(self.but_1)

        # Update a widget property.
        self.label_result.text = 'loading' #''Classificaition in progress'
        # self.label_result.text = ('The UI remains responsive while the '
        #                    'second thread is running.')

        # Create and add a new widget.
        anim_bar = Factory.AnimWidget()
        self.layout_pie_chart.add_widget(anim_bar)

        # Animate the added widget.
        anim = Animation(opacity=0.3, width=100, duration=0.6)
        anim += Animation(opacity=1, width=400, duration=0.8)
        anim.repeat = True
        anim.start(anim_bar)

    @mainthread
    def update_label_text(self, new_text):
        # pass
        self.label_info.text = new_text

    @mainthread
    def show_classification_result(self, iLabel, lstFinalPercentages):
        # self.label_result.text = ('Second thread exited, a new thread has started. '
        #                    'Close the app to exit the new thread and stop '
        #                    'the main process.')
        # self.label_result.text = 'loading done'

        # self.label_info.text = str(int(self.label_info.text) + 1)
        # print('show result')
        self.layout_pie_chart.clear_widgets()

        if iLabel is not None:
            self.renderPlotChar(lstFinalPercentages)
            self.label_result.text = 'Result: ' + CategoryStr.lstStrCategories[iLabel]
            print('iLabel: ', iLabel)
            print('lstFinalPercentages: ', lstFinalPercentages)
        else:
            self.label_result.text = 'Result: '

        self.button_classifier.disabled = False                      # re-enable button

    # self.remove_widget(self.layout_pie_chart)

    def infinite_loop(self):
        iteration = 0
        while True:
            if self.stop.is_set():
                # Stop running this thread so the main Python process can exit.
                return
            iteration += 1
            print('Infinite loop, iteration {}.'.format(iteration))
            time.sleep(1)

    def show_wordcloud(self, text, label):

        print('text: ', text)
        img = (Image.open(self.strPath + "/media/icons/" + CategoryStr.lstStrIcons[label])).split()[-1]
        print(label)
        # the mask is inverted, so invert it again
        img = ImageOps.invert(img)
        img = img.resize((512, 512), Image.NONE)
        imgMask = np.array(img)

        wordcloud = WordCloud(background_color=(48, 48, 48), mask=imgMask).generate(text)
        self.layout_diagram1.clear_widgets()
        plt.figure(2)
        plt.imshow(wordcloud)
        plt.axis("off")
        fig = plt.gcf()
        fig.patch.set_facecolor((48/255, 48/255, 48/255))
        self.layout_diagram1.add_widget(FigureCanvas(fig))

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
        self.repoClassifier = RepositoryClassifier(bUseStringFeatures=False)
        self.repoClassifier.loadModelFromFile()

        self.strPath = os.path.dirname(__file__)

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

        url_in = "".join(self.textfield_input.text.split())               # read input and remove whitespaces
        self.textfield_input.text = url_in
        print("[INFO] Starting Process with \"" + url_in + "\"")    # print info to console
        valid = self.validate_url(url_in)                           # validate input and handle Errors

        if valid:
            self.button_classifier.disabled = True  # disable button
            self.start_classification_thread(self.label_info.text, url_in)

    def renderPlotChar(self, lstFinalPercentages):
        print("render piechart")
        # self.log_console.text = ""                                      # clear console
        self.log_console.scroll_y = 0                             # makes the console scroll down automatically
        # for i in range(0, 50):                                          # demonstrate console
            # self.log_console.text += ("Button pressed, " + str(i) + "\n")

        # ADDING A PIE CHART!
        # The slices will be ordered and plotted counter-clockwise.
        labels = CategoryStr.lstStrCategories

        # multiplicate every element with 100
        lstFinalPercentages[:] = [x * 100 for x in lstFinalPercentages]

        lstLabelsPieChart = [None] * len(labels)
        lstLabelsPieChart = labels[:]

        for i, _ in enumerate(labels):
            lstLabelsPieChart[i] += ' (' + str(round(lstFinalPercentages[i], 1)) + '%)'

        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'gray', 'lightblue', 'tomato']

        # http://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        iMaxIndex = lstFinalPercentages.index(max(lstFinalPercentages))

        lstExplode = [0] * len(lstFinalPercentages)
        lstExplode[iMaxIndex] = 0.1
        explode = lstExplode #(0, 0, 0.1, 0)  # only "explode" the 1st slice (i.e. 'Dogs')
        # self.iFigIndex += 1
        fig = plt.figure(1, figsize=(10, 10), dpi=70)
        fig.clear()

        # patches, texts = plt.pie(lstFinalPercentages, explode=explode, colors=colors, # labels=labels, autopct='%1.1f%%', shadow=True,
        #           startangle=90)
        plt.pie(lstFinalPercentages, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%', shadow=True,
                  startangle=90)

        # plt.axis('equal')                                        # this was the actual cause of the resizing !!!
        #  -> this causes a warning; alternative us fig,set_tight_layout(True)
        # plt.tight_layout()                                         # http://matplotlib.org/users/tight_layout_guide.html

        # plt.legend(patches, lstLabelsPieChart, loc=(0.97, 0.3), prop={'size':10})

        fig = plt.gcf()

        # an alternative to get a round pie-chart is to use .set_aspect(1)
        # http://stackoverflow.com/questions/8418566/how-to-draw-a-round-pie-in-non-square-figure-size-in-matplotlib-python
        ax = plt.gca()
        ax.set_aspect(1)
        fig.set_tight_layout(True)

        # fig.patch.set_facecolor('1')
        # fig.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.3)

        # plt.show()

        self.layout_pie_chart.clear_widgets()
        self.layout_pie_chart.add_widget(FigureCanvas(fig))
        # fig.clear()

        # self.layout_pie_chart.add_widget(fig)


class RepositoryClassifierApp(App):
    icon = 'logo_small.png'                          # change window icon

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        self.root.stop.set()


    def build(self):
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        # Window.size = (1200, 800)

        layGUI = GUILayout()
        layGUI.initialize()

        return layGUI


gui = RepositoryClassifierApp()
gui.run()

