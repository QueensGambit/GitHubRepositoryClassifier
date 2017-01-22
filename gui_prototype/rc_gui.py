"""
@file: gui_protoype.py
Created on 07.01.2017 23:06
@project: GitHubRepositoryClassifier

@author: Lukas

GUI Prototype using kivy
Presenting an easily usable GUI to the user to make classification as easy as possible.
Also includes some visuals like multiple plots and a word cloud as well as some examples

"""
import kivy
from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
# Config.set('graphics', 'resizable', False)

# This block is needed to use the right dll-file for building
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kivy.require("1.9.0")

import ctypes
import os
import sys

if getattr(sys, 'frozen', False):
  # Override dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW('G:/Program Files/Anaconda3/Library/bin')
  # Init code to load external dll
  ctypes.CDLL('mkl_avx2.dll')
  ctypes.CDLL('mkl_def.dll')
  ctypes.CDLL('mkl_vml_avx2.dll')
  ctypes.CDLL('mkl_vml_def.dll')

  # Restore dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib
# matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
matplotlib.use('module://lib.kivy.garden.matplotlib.backend_kivy')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kivy.uix.label import Label
from lib.kivy.garden.matplotlib.backend_kivyagg import FigureCanvas     # don't worry, it works even though its red
from kivy.uix.popup import Popup
import clipboard
from sklearn import decomposition
import matplotlib.patches as mpatches

import sys, os
import matplotlib.pyplot as plt
import pylab as pl
# from colour import Color

# add the current directory to the system path in order to find the modules in relative path
# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.abspath("../prototype"))


from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs.io_agent import InputOutputAgent  # this import is used to enable or disable the GithubToken

from prototype.definitions.categories import CategoryStr
import webbrowser

# threading and animation
# multithreading in kivy:
# https://github.com/kivy/kivy/wiki/Working-with-Python-threads-inside-a-Kivy-application
import threading
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.factory import Factory

from lib.wordcloud import WordCloud, ImageColorGenerator

from PIL import Image
import numpy as np
from pathlib import Path

kivy.require("1.9.0")


# http://stackoverflow.com/questions/2297933/creating-a-custom-sys-stdout-class
# other options:
# - redirect_stdout
# - contextlib
# - overload print() function
# ...

class StaticVars:
    b_api_checkbox_state = False                    # checkbox status for global use
    b_checkbox_download = False                    # checkbox status for global use
    b_run_loading = True                            # boolean to stop loading thread from running too late
    animation_loading = Animation()
    anim_bar = None
    str_stdOutPuffer = ""
    fig_pie_chart = None


class StdOut(object):
    def __init__(self, log_console, oldStdOut):
        # self.txtctrl = txtctrl
        self.log_console = log_console
        self.oldStdOut = oldStdOut
        # see more informations about thread-synchronisation
        # https://docs.python.org/3/library/asyncio-sync.html
        self.lock = threading.Lock()

    def write(self, string):
        # https://docs.python.org/3/library/asyncio-sync.html
        try:
            self.lock.acquire()
            self.oldStdOut.write(string)
            StaticVars.str_stdOutPuffer += string
        finally:
            self.lock.release()

    def flush(self):
        pass


class Radar(object):
    backgroundcolor = (48 / 255, 48 / 255, 48 / 255) #'w'
    # http://stackoverflow.com/questions/4804005/matplotlib-figure-facecolor-background-color
    plt.rcParams['axes.facecolor'] = backgroundcolor
    plt.rcParams['savefig.facecolor'] = backgroundcolor
    plt.rcParams['axes.edgecolor'] = 'silver' #''w'
    plt.rcParams['axes.labelcolor'] = 'silver' #''w'
    plt.rcParams['grid.color'] = 'silver' #''w'
    # plt.rcParams['legend.facecolor'] = 'w'

    def __init__(self, fig, titles, labels, color='b', rect=None):
        if rect is None:
            # rect = [0.05, 0.05, 0.95, 0.95]
            # set a constant offset and scale
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(titles)
        self.color = color
        self.angles = np.arange(45, 45+360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                         for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14, color=self.color)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 5), angle=angle, labels=label, backgroundcolor=self.backgroundcolor)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


class InfoPopup(Popup):
    """
    The Information Popup which is called by show_info() in the GUILayout class.
    """
    pass


class SettingsPopup(Popup):
    """
    The Settings Popup which is called by show_settings() in the GUILayout class.
    """
    checkbox_api_token = ObjectProperty()                   # The checkbox to toggle the usage of the API Token
    label_api_error = ObjectProperty()                      # The Label to Output potential errors
    checkbox_download = ObjectProperty()                    # The checkbox to toggle the redownload
    label_download_error = ObjectProperty()                 # The Label to Output potential errors

    def __init__(self, windowParent):
        """
        Called upon opening of the Settings Popup
        Override the active state of the API checkbox to display the current internal saved state

        :param windowParent: windows handle of the main frame where the console is located
        """
        super(SettingsPopup, self).__init__()
        self.checkbox_api_token.active = StaticVars.b_api_checkbox_state
        self.checkbox_download.active = StaticVars.b_checkbox_download
        self.windowParent = windowParent

    def switch_api(self, b_status):
        """
        Called by the API Checkbox in the Settings Popup.
        checks whether the Token is is valid/existent and switches accordingly.

        :param b_status: provides the current state of the checkbox
        :return:
        """
        try:
            self.label_api_error.text = ""                              # Reset Error Label
            InputOutputAgent.setWithToken(b_status)                     # set token if possible
            StaticVars.b_api_checkbox_state = b_status                  # save state of the token
            print('[INFO] Use API updated to: ' + str(b_status))        # print info to console
        except ConnectionError as ce:
            self.label_api_error.text = "[ERROR] No Connection could be established."
            print("[ERROR] No Connection could be established: " + str(ce))
            self.checkbox_api_token.active = False                      # update checkbox to display the internal state
            StaticVars.b_api_checkbox_state = False                     # set the internal state to false
        self.windowParent.update_console()                              # update the window console

    def switch_download(self, b_status):
        """
        Called by the Download Checkbox in the Settings Popup.
        switches accordingly

        :param b_status: provides the current state of the checkbox
        :return:
        """
        try:
            self.label_download_error.text = ""                              # Reset Error Label
            InputOutputAgent.setRedownload(b_status)                         # Setting the redownlad on the IOAgent
            StaticVars.b_checkbox_download = b_status                       # save state of the token
            print('[INFO] Re-Download updated to: ' + str(b_status))        # print info to console
        except Exception as ce:
            self.label_download_error.text = str(ce)
            print("[ERROR] " + str(ce))
            self.checkbox_download.active = False                      # update checkbox to display the internal state
            StaticVars.b_checkbox_download = False                     # set the internal state to false
        self.windowParent.update_console()                              # update the window console


class MultiDimBackground:

    def __init__(self, lstTrainData, centroids_):
        # used for plot multi_dim_data
        self.data2d = None
        self.pca = None
        self.centroids2d = None

        if lstTrainData.shape[1] > 2:
            self.pca = decomposition.PCA(n_components=2)
            self.data2d = self.pca.fit_transform(lstTrainData)
            self.centroids2d = self.pca.transform(centroids_)


        # calculate the clusters
        n_clusters = 7
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kmeans.fit(self.data2d)
        h = .02

        x_min, x_max = self.data2d[:, 0].min() - 1, self.data2d[:, 0].max() + 1
        y_min, y_max = self.data2d[:, 1].min() - 1, self.data2d[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        self.Z = kmeans.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        self.Z = self.Z.reshape(self.xx.shape)


class GUILayout(BoxLayout):
    """
    The Main Layout of the Main Window. Handles most events
    """

    stop = threading.Event()

    # define the ObjectProperties to communicate with the .kv file
    textfield_input = ObjectProperty()                    # user input line
    button_classifier = ObjectProperty()            # the button above the user input. should start the process
    label_info = ObjectProperty()                   # the label underneath the user input. short descriptions go here
    log_console = ObjectProperty()                  # console to output logs. just add text to it.
    label_result = ObjectProperty()                 # the big Result Label in the result corner
    label_second_result = ObjectProperty()          # the small, secondary Result Label under the big Result Label
    layout_pie_chart = ObjectProperty()             # the Layout for the piechart in the result corner
    layout_diagram1 = ObjectProperty()              # the three TabbedPanelItems to put a diagram, expand if needed
    layout_diagram2 = ObjectProperty()              # ↑
    layout_diagram3 = ObjectProperty()              # ↑

    # def initialize(self):
    def __init__(self):
        """
        Initializes the Main Layout (GUILayout), is called after the its creation in the RepositoryClassifierApp

        :return:
        """
        super().__init__()

        self.log_console.scroll_y = 0                             # makes the console scroll down automatically

        # initialize the repositoryClassifier
        self.repoClassifier = RepositoryClassifier(bUseStringFeatures=True)  #bUseStringFeatures=False
        self.clf, self.lstMeanValues, self.matIntegerTrainingData, self.lstTrainLabels, self.lstTrainData, self.normalizer, self.normalizerIntegerAttr,_ = self.repoClassifier.loadModelFromFile()

        # self.strPath = os.path.dirname(__file__)
        self.strPath = str(Path())

        self.log_console.scroll_y = 0                                   # makes the console scroll down automatically

        self.multiDimBackground = MultiDimBackground(self.lstTrainData, self.clf.centroids_)

    def initialize_std_out_redirection(self):
        oldStdOut = sys.stdout
        # overload load the sys.strdout to a class-instance of StdOut
        sys.stdout = StdOut(self.log_console, oldStdOut)
        self.init_plotting()

    @mainthread
    def update_console(self):

        self.log_console.text += StaticVars.str_stdOutPuffer
        StaticVars.str_stdOutPuffer = ""

    @mainthread
    def reset_result_layout(self):
        """
        gets called whenever some operation failed to allow for another try.

        :return:
        """
        StaticVars.b_run_loading = False
        self.layout_pie_chart.clear_widgets()
        self.label_result.text = "ERROR"
        self.label_second_result.text = ""
        self.button_classifier.disabled = False
        self.update_console()
        StaticVars.animation_loading.cancel(StaticVars.anim_bar)

    # threading
    @mainthread
    def start_classification_thread(self, l_text, url_in):
        """
        start the classification Thread to run parallel to the GUI so it doesn't freeze in the meantime

        :param l_text: apparently needs this to not break. can be anything, isn't used anyway
        :param url_in: the URL to check in for the Repository
        :return:
        """
        threading.Thread(target=self.classification_thread, args=(l_text, url_in), daemon=True).start()

    def classification_thread(self, l_text, url_in):
        """
        Tries to reach the Repository and start the classification process.
        If successful starts the rendering of the different visualisations

        :param l_text: apparently needs this to not break. can be anything, isn't used anyway
        :param url_in: the URL to check in for the Repository
        :return:
        """

        # Remove a widget, update a widget property, create a new widget,
        # add it and animate it in the main thread by scheduling a function
        # call with Clock.
        Clock.schedule_once(self.start_loading_animation, 0)

        try:
            iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, self.lstNormedInputFeatures = self.repoClassifier.predictCategoryFromURL(url_in)
            # print('self.lstNormedInputFeatures: ', self.lstNormedInputFeatures[:4])

            # Remove some widgets and update some properties in the main thread
            # by decorating the called function with @mainthread.
            self.show_classification_result(iLabel, iLabelAlt, lstFinalPercentages, tmpRepo)

        except ConnectionError as ce:
            print("[ERROR] A connection error occurred: " + str(ce))
            self.set_error("[ERROR] The Repository is not accessible")
            self.reset_result_layout()
        except IndexError as ie:
            print("[ERROR] An Index Error occurred: " + str(ie))
            self.set_error("[ERROR] Have you added the repository?")
            self.reset_result_layout()
        except Exception as ex:
            print("[ERROR] An unknown Error occurred: " + str(ex))
            self.set_error("[ERROR] An unknown Error occurred")
            self.reset_result_layout()

    @mainthread
    def start_loading_animation(self, *args):
        """
        Creates the User-Feedback while loading, such as setting the label_error and showing the loading animation.

        :param args:
        :return:
        """
        # Remove the button.
        self.layout_pie_chart.clear_widgets()
        # self.remove_widget(self.but_1)

        # Update a widget property.
        self.set_info("[INFO] Classification in progress")
        self.label_result.text = "Loading..."
        self.label_second_result.text = ""

        # Create and add a new widget.
        StaticVars.anim_bar = Factory.AnimWidget()

        # Animate the added widget.
        StaticVars.animation_loading = Animation(opacity=0.3, width=100, duration=0.6)
        StaticVars.animation_loading += Animation(opacity=1, width=400, duration=0.8)
        StaticVars.animation_loading.repeat = True
        StaticVars.animation_loading.start(StaticVars.anim_bar)

        self.layout_pie_chart.add_widget(StaticVars.anim_bar)

    @mainthread
    def init_plotting(self):
        iNumOfFigures = 5
        for i in range(1, iNumOfFigures+1):
            plt.figure(i)

    @mainthread
    def update_pie_chart(self, fig): #*args):
        self.layout_pie_chart.clear_widgets()
        self.layout_pie_chart.add_widget(FigureCanvas(fig))

    @mainthread
    def update_result_label(self, iLabel, iLabelAlt, lstFinalPercentagesSorted):

        if lstFinalPercentagesSorted[5] > lstFinalPercentagesSorted[6] - .5:
            self.label_second_result.text = "Secondary Result: " + CategoryStr.lstStrCategories[iLabelAlt]

        self.label_result.text = 'Result: ' + CategoryStr.lstStrCategories[iLabel]

    @mainthread
    def enable_classification(self):
        StaticVars.b_run_loading = False
        StaticVars.animation_loading.cancel(StaticVars.anim_bar)
        self.button_classifier.disabled = False                      # re-enable button
        self.update_console()

    @mainthread
    def update_result_label_no_result(self):
        self.label_result.text = 'No Result'
        self.label_second_result = ""

    @mainthread
    def update_wordcloud(self, fig):
        self.layout_diagram1.clear_widgets()
        self.layout_diagram1.add_widget(FigureCanvas(fig))

    @mainthread
    def update_no_wordcloud(self):
        self.layout_diagram1.clear_widgets()
        self.layout_diagram1.add_widget(Label(text="The Repository doesn't contain any words"))

    @mainthread
    def update_multi_dim(self, fig):
        self.layout_diagram3.clear_widgets()
        self.layout_diagram3.add_widget(FigureCanvas(fig))

    @mainthread
    def update_plot_net_diagram(self, fig):
        self.layout_diagram2.clear_widgets()
        self.layout_diagram2.add_widget(FigureCanvas(fig))

    def show_classification_result(self, iLabel, iLabelAlt, lstFinalPercentages, tmpRepo):  #l_text,
        """
        Creates the user output for the final result:
        The pie chart as well as the label in the top right corner

        :param iLabel: index of the Category Enum of the found result
        :param iLabelAlt: index of the Category Enum of the secondary result
        :param lstFinalPercentages: List of the percentages each Category yields
        :param tmpRepo: a Repository object
        :return:
        """

        if iLabel is not None:
            # pie chart
            self.update_pie_chart(self.render_pie_chart(lstFinalPercentages))

            lstFinalPercentages.sort()

            self.update_result_label(iLabel, iLabelAlt, lstFinalPercentages)

            self.set_info("[INFO] Classification complete")

            # Wordcloud
            dicFoundWords = tmpRepo.getDicFoundWords()
            strText = str(tmpRepo.getFilteredReadme(bApplyStemmer=True, bCheckStopWords=True) + " " + tmpRepo.getFilteredRepoDescription(
                bApplyStemmer=True, bCheckStopWords=True))

            if not strText.isspace():
                self.update_wordcloud(self.show_wordcloud(strText, iLabel, dicFoundWords, tmpRepo.getRepoLanguage()))

            else:
                self.update_no_wordcloud()

            lstCurIntegerFeatures = tmpRepo.getNormedFeatures(lstMeanValues=self.lstMeanValues)

            self.update_multi_dim(self.plot_multi_dim(lstCurIntegerFeatures))

            # net diagram
            self.update_plot_net_diagram(self.plot_net_diagram(tmpRepo, iLabel))

        else:
            self.update_result_label_no_result()

        self.enable_classification()

    def show_wordcloud(self, text, iLabel, dicFoundWords, strRepoLanguage):
        """
        Creates the Wordcloud in the first Diagram Tab.

        :param text: The Text to create the word cloud from
        :param iLabel: index of the Category Enum of the found result
        :return:
        """

        img = np.array((Image.open(self.strPath + "/media/icons/colored/" + CategoryStr.lstStrIcons[iLabel])))

        # create coloring from image
        img_colors = ImageColorGenerator(img)
        # wordcloud = WordCloud(background_color=(48, 48, 48), mask=imgMask).generate(text)
        wordcloud = WordCloud(background_color=(48, 48, 48), mask=img, color_func=img_colors, max_words=2000).generate(text)
        # self.layout_diagram1.clear_widgets()
        plt.figure(2)
        plt.imshow(wordcloud)

        if dicFoundWords.keys() is not None:
            labels = [l for l in dicFoundWords.keys()]
            iMaxNumberLabels = 12
            labels = [int(x) for i, x in enumerate(labels) if i < iMaxNumberLabels]
            #http://stackoverflow.com/questions/3940128/how-can-i-reverse-a-list-in-python
            labelsRev = labels[::-1] #list(reversed(labels))
            # positions = [(2, 5), (1, 1), (4, 8)]
            descriptions = [v for i, v in enumerate(dicFoundWords.values()) if i < iMaxNumberLabels]

            descriptionsRev = descriptions[::-1] #

            if len(dicFoundWords.keys()) > iMaxNumberLabels:
                labelsRev.append(0)
                descriptionsRev.append("...")

            # Create a legend with only labels
            # http://stackoverflow.com/questions/28739608/completely-custom-legend-in-matplotlib-python
            # TODO: decide whether to put this in or not
            proxies = [self.create_proxy(item) for item in labelsRev]
            plt.rcParams['text.color'] = CategoryStr.lstStrColors[iLabel] #'silver'

            ax = plt.gca()
            # ax.get_legend().get_title()
            strTitle = 'language: ' + strRepoLanguage + '\n'
            strTitle += 'found words in vocab:'
            ax.legend(proxies, descriptionsRev, numpoints=1, markerscale=2, loc=(1.3,0.0), title=strTitle)#loc='upper right'
            plt.rcParams['text.color'] = 'black'

        plt.axis("off")

        fig = plt.gcf()
        fig.patch.set_facecolor((48/255, 48/255, 48/255))
        return fig

    def create_proxy(self, label):
        line = matplotlib.lines.Line2D([0], [0], linestyle='none', color='silver', mfc='silver', #mfc='black',
                                       mec='none', marker=r'$\mathregular{{{}}}$'.format(label))
        return line

    def open_repo_url(self):
        """
        opens the current url in the preferred web-browser in case if the valid check was OK

        :return:
        """
        str_url_in = self.textfield_input.text
        if self.validate_url(str_url_in) is True:
            webbrowser.open(str_url_in)

    def open_github_project_url(self):
        """
        opens the url of the github source code of the project

        :return:
        """
        webbrowser.open("https://github.com/QueensGambit/GitHubRepositoryClassifier")

    def show_info(self):
        """
        Displays the info popup, called by the ActionBar

        :return:
        """
        info_popup = InfoPopup()
        info_popup.open()

    def show_documentation(self):
        """
        Opens the sphinx-code-documentation in a web browser, called by the ActionBar

        :return:
        """
        webbrowser.open("https://github.com/QueensGambit/GitHubRepositoryClassifier/wiki/Documentation")

    def show_settings(self):
        """
        Displays the settings popup, called by the ActionBar

        :return:
        """
        settings_popup = SettingsPopup(windowParent=self)
        settings_popup.open()

    def save_log(self):
        """
        Displays the save file popup, called by the Button underneath the Console

        :return:
        """
        save_popup = FileSaverPopup(self)
        save_popup.log_text = self.log_console.text
        save_popup.open()

    def paste(self):
        """
        Pastes the content of the clipboard into the url input textfield,
        called by the Paste Button next to the Textfield

        :return:
        """
        # get clipboard data
        self.textfield_input.text = clipboard.paste()
        # print('paste-button pressed')
        # print(clipboard.paste())
        print('[INFO] pasted text:', clipboard.paste())
        self.update_console()

    def validate_url(self, url_in):
        """
        Performs some simple string checks to validate the URL for further processing

        :param url_in: The URL to perform the checks on
        :return:
        """
        if url_in == "":
            print("[ERROR] Input is empty")
            self.set_error("[ERROR] Input is empty")
            self.update_console()
            return False
        elif not url_in.startswith("https://"):
            print("[ERROR] Input doesn't start with https://")
            self.set_error("[ERROR] Input doesn't start with https://")
            self.update_console()
            return False
        elif not url_in.startswith("https://github.com/"):
            print("[ERROR] Input is not a GitHub URL")
            self.set_error("[ERROR] Input is not a GitHub URL")
            self.update_console()
            return False
        else:
            print("[INFO] Input is a valid URL")
            self.set_info("[INFO] Input is a valid URL")
            self.update_console()
            return True

    @mainthread
    def set_info(self, info):
        """
        put the info text as info text, text color to white

        :param info: the text to display
        :return:
        """
        self.label_info.color = 1, 1, 1, 1
        self.label_info.text = info

    @mainthread
    def set_error(self, error):
        """
        put the info text as error text, text color to red

        :param error:
        :return:
        """
        self.label_info.color = 1, 0, 0, 1
        self.label_info.text = error

    @mainthread
    def classify_button_pressed(self):
        """
        Gets called from the "Classify" button, starts the classification thread function

        :return:
        """

        url_in = "".join(self.textfield_input.text.split())               # read input and remove whitespaces
        self.textfield_input.text = url_in
        print("[INFO] Starting Process with \"" + url_in + "\"")        # print info to console
        valid = self.validate_url(url_in)                               # validate input and handle Errors
        if valid is True:
            self.button_classifier.disabled = True                      # disable button
            StaticVars.b_run_loading = True                             # enable loading screen
            self.start_classification_thread(self.label_info.text, url_in)

    def render_pie_chart(self, lstFinalPercentages):
        """
        Creates the pie chart

        :param lstFinalPercentages: the percentages to use in the piechart
        :return:
        """
        # The slices will be ordered and plotted counter-clockwise.
        labels = CategoryStr.lstStrCategories

        # multiplicate every element with 100
        lstFinalPercentages[:] = [x * 100 for x in lstFinalPercentages]

        lstLabelsPieChart = labels[:]

        for i, _ in enumerate(labels):
            lstLabelsPieChart[i] += ' (' + str(round(lstFinalPercentages[i], 1)) + '%)'

        # http://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        iMaxIndex = lstFinalPercentages.index(max(lstFinalPercentages))

        lstExplode = [0] * len(lstFinalPercentages)
        lstExplode[iMaxIndex] = 0.1
        explode = lstExplode                                            # only "explode" the biggest slice
        fig = plt.figure(1, figsize=(10, 10), dpi=70)                   # this causes an error when multi-threading is active and it wasn't intialized via the main thread
        #  <kivy.uix.gridlayout.GridLayout object at 0x000000D503F5F8D0> have no cols or rows set, layout is not triggered.
        fig.clear()

        plt.pie(lstFinalPercentages, explode=explode, colors=CategoryStr.lstStrColors, labels=labels, autopct='%1.1f%%', shadow=True,
                startangle=90)

        fig = plt.gcf()

        # an alternative to get a round pie-chart is to use .set_aspect(1)
        # http://stackoverflow.com/questions/8418566/how-to-draw-a-round-pie-in-non-square-figure-size-in-matplotlib-python
        ax = plt.gca()
        ax.set_aspect(1)
        fig.set_tight_layout(True)

        # fig.patch.set_facecolor('1')
        # fig.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.3)

        return fig

    def load_example(self, link):
        """
        write the example link into the textfield, gets called by the Examples in the Action Bar

        :param link: the link to the repository, without the github in front!
        :return:
        """
        self.textfield_input.text = "https://github.com/" + link

    def plot_multi_dim(self, lstCurIntegerFeatures):
        """
        show multidimensional data in 2D plot
        :return:
        """
        bPlotAllFeatures = True

        if bPlotAllFeatures is True:
            clf = self.clf
            lstTrainLabels = self.lstTrainLabels

            if len(self.lstTrainData) < 2:
                raise Exception('Lenght of array >= 2')

            fig = plt.figure(3)
            plt.cla()
            plt.clf()

            # plot the contours of kmeans
            plt.imshow(self.multiDimBackground.Z, interpolation='nearest',
                       extent=(self.multiDimBackground.xx.min(), self.multiDimBackground.xx.max(), self.multiDimBackground.yy.min(), self.multiDimBackground.yy.max()),
                       cmap=plt.cm.Paired, alpha=0.1,
                       aspect='auto', origin='lower')

            lstColors = [None] * len(lstTrainLabels)

            rect = [0.05, 0.05, 0.9, 0.9]

            # set a ceratin aspect ratio
            ax = plt.gca()
            ax.set_aspect(0.6)#'equal')

            plt.axis("on")

            for i, iLabel in enumerate(lstTrainLabels):
                lstColors[i] = CategoryStr.lstStrColors[iLabel]

            plt.scatter(self.multiDimBackground.data2d[:, 0], self.multiDimBackground.data2d[:, 1], cmap=plt.cm.Paired, color=lstColors, alpha=0.3)

            # plot the centroid
            handleCentroids = plt.scatter(self.multiDimBackground.centroids2d[:, 0], self.multiDimBackground.centroids2d[:, 1],
                                          marker='x', s=200, linewidths=3,  #169 marker = x d D
                        color=CategoryStr.lstStrColors, zorder=10) #edgecolor='black',

            # plot the current sample via the given integer features
            ptCurRepo = self.multiDimBackground.pca.transform(self.lstNormedInputFeatures)

            handleCurRepo = plt.scatter(ptCurRepo[:, 0], ptCurRepo[:, 1],
                        marker='*', s=400, linewidths=3,
                        color='white', zorder=10)  #gold violet red , linewidth='3', edgecolor='white',

            # set the xlim and ylim to a custom value
            # http://stackoverflow.com/questions/11400579/pyplot-zooming-in
            fMaxVal = 0.4

            xmin = -fMaxVal
            xmax = fMaxVal
            ymin = -fMaxVal
            ymax = fMaxVal
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xticks(())
            plt.yticks(())

            # create a custom legend in color
            lstPatches = [None] * len(CategoryStr.lstStrCategories)
            for i, strCategory in enumerate(CategoryStr.lstStrCategories):
                lstPatches[i] = mpatches.Patch(color=CategoryStr.lstStrColors[i], label=strCategory)

            leg = plt.legend(handles=lstPatches, loc=(1.0,0.0))

            for i, text in enumerate(leg.get_texts()):
                text.set_color(CategoryStr.lstStrColors[i])

            plt.rcParams['text.color'] = 'silver'

            plt.legend((handleCentroids, handleCurRepo),
                       ('Centroids', 'Current Repository'),
                       scatterpoints=1,
                       loc='lower left',
                       ncol=3,
                       fontsize=10)
            plt.rcParams['text.color'] = 'black'

            plt.gca().add_artist(leg)

            fig = plt.gcf()
            fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))

            return fig
        else:
            clf = self.clf
            lstTrainLabels = self.lstTrainLabels
            data = self.matIntegerTrainingData

            normalizer = preprocessing.Normalizer()
            normalizer = normalizer.fit(data)
            data = normalizer.transform(data)
            print('data:', data)

            # if len(data) < 2:
            if len(self.lstTrainData) < 2:
                raise Exception('Lenght of array >= 2')

            pca = None
            if self.lstTrainData.shape[1] > 2:
                pca = decomposition.PCA(n_components=2)
                data = pca.fit_transform(data)

            n_clusters = 7
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            kmeans.fit(data)
            h = .02

            x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
            y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig = plt.figure(3)
            plt.cla()
            plt.clf()

            lstColors = [None] * len(lstTrainLabels)

            rect = [0.05, 0.05, 0.9, 0.9]

            # set a ceratin aspect ratio
            ax = plt.gca()
            ax.set_aspect(0.6)  # 'equal')

            for i, iLabel in enumerate(lstTrainLabels):
                lstColors[i] = CategoryStr.lstStrColors[iLabel]

            plt.scatter(data[:, 0], data[:, 1], cmap=plt.cm.Paired, color=lstColors, alpha=0.5)

            # plot the centroid

            # plot the current sample via the given integer features
            # lstCurIntegerFeatures = self.normalizer.transform(lstCurIntegerFeatures)
            print('lstCurIntegerFeatures:', lstCurIntegerFeatures)
            lstCurIntegerFeatures = normalizer.transform(lstCurIntegerFeatures)
            ptCurRepo = pca.transform(lstCurIntegerFeatures)

            plt.scatter(ptCurRepo[:, 0], ptCurRepo[:, 1],
                        marker='*', s=400, linewidths=3,
                        color='violet', zorder=10)  # gold

            # set the xlim and ylim to a custom value
            # http://stackoverflow.com/questions/11400579/pyplot-zooming-in
            # fMaxVal = 0.4
            fMaxVal = 1.4

            xmin = -fMaxVal
            xmax = fMaxVal
            ymin = -fMaxVal
            ymax = fMaxVal
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xticks(())
            plt.yticks(())

            # create a custom legend in color
            lstPatches = [None] * len(CategoryStr.lstStrCategories)
            for i, strCategory in enumerate(CategoryStr.lstStrCategories):
                lstPatches[i] = mpatches.Patch(color=CategoryStr.lstStrColors[i], label=strCategory)

            leg = plt.legend(handles=lstPatches, loc=(1.0, 0.0))  # 'right')

            for i, text in enumerate(leg.get_texts()):
                text.set_color(CategoryStr.lstStrColors[i])

            fig = plt.gcf()
            fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))

            return fig

    @mainthread
    def plot_barchart(self, lsIntegerAttributes):
        """
        function to plot a barchart using the integer values. not used anymore.

        :param lsIntegerAttributes:
        :return:
        """
        plt.figure(4)
        lsIntegerAttribute = lsIntegerAttributes[0][:4]
        iN = len(lsIntegerAttribute)
        ind = np.arange(iN)  # the x locations for the groups
        width = 0.15  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, lsIntegerAttribute, width, color='r')  # yerr=menStd)

        # add some text for labels, title and axes ticks
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('Subscriber', 'Open Issues', 'DevTime', 'Size'))

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        autolabel(rects1)

        fig = plt.gcf()
        fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))

        self.layout_diagram2.clear_widgets()
        self.layout_diagram2.add_widget(FigureCanvas(fig))

    def plot_net_diagram(self, repo, iLabel):
        """
        function to plot a net diagram to display the Integer Attributes.

        :param repo: The Repository to get the values from
        :param iLabel: The Result of the Classification
        :return:
        """
        # http://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes
        # import pylab as pl -> moved to the top
        lsAttributes = self.lstNormedInputFeatures[0][:4]

        fig = plt.figure(5, figsize=(0.1, 0.1), dpi=80)

        fig.clear()

        lstIntegerFeaturesRaw = repo.getIntegerFeatures()

        titles = ['Subscribers\n' + str(lstIntegerFeaturesRaw[0]),
                  'Open Issues\n' + str(lstIntegerFeaturesRaw[1]),
                  'DevTime\n' + str(lstIntegerFeaturesRaw[2]),
                  'Size\n' + str(lstIntegerFeaturesRaw[3])]


        labels = [
            [],
            [],
            [],
            []
        ]

        radar = Radar(fig, titles, labels, color='silver')  # color=CategoryStr.lstStrColors[iLabel]
        #http://stackoverflow.com/questions/24076297/how-to-truncate-a-string-using-str-format-in-python

        iMaxStrLen = 33

        # strRepoLongName = repo.getUser() + '/' + repo.getName()
        strRepoLongName = repo.getUser() + '\n' + repo.getName()

        radar.plot(lsAttributes * 10, "-", lw=2, color=CategoryStr.lstStrColors[iLabel], alpha=0.4, label=strRepoLongName) #"This Repo")

        '{:.5}'.format('aaabbbccc')

        leg = radar.ax.legend(loc=(1, .6))

        # http: // stackoverflow.com / questions / 13828246 / matplotlib - text - color - code - in -the - legend - instead - of - a - line
        # set the color to the category
        for i, text in enumerate(leg.get_texts()):
            text.set_color(CategoryStr.lstStrColors[iLabel])

        fig = plt.gcf()
        fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))

        return fig

class FileSaverPopup(Popup):
    """
    The Popup to save the console output to a log file. called by save_log() in the GUILayout class.
    """
    label_save = ObjectProperty()           # the label to output potential error messages in the File Saver Popup
    file_chooser = ObjectProperty()
    def __init__(self, windowParent):
        super(FileSaverPopup, self).__init__()
        self.windowParent = windowParent
        strExportPath = os.path.expanduser('~')
        print('strExportPath: ', strExportPath)
        self.file_chooser.path = os.path.expanduser('~')

    def save_file(self, path, filename):
        """
        Called by the "save" button in the Popup.
        Saves the file with the input filename to the path. Handles Permission Errors.

        :param path: The current folder path to save the file into, ends with "/"
        :param filename: The filename to save the file as
        :return:
        """
        stream = None

        try:
            with open(os.path.join(path, filename), 'w') as stream:
                stream.write(self.log_text)
                stream.close()
            self.dismiss()
            print("[INFO] Logfile saved to: " + path + "\\" + filename)
        except PermissionError as err:
            print("[ERROR] Logfile couldn't be saved. Permission denied. Path: " + path + "\nError: " + str(err))
            self.label_save.text = "[ERROR] Couldn't save. Permission denied."
            if stream is not None:
                stream.close()

        self.windowParent.update_console()


class RepositoryClassifierApp(App):
    """
    The Main App Class that contains the App
    """
    icon = 'logo_small.png'                          # change window icon

    def on_stop(self):
        """
        gets called when the main Kivy Event Loop is about to stop, setting a stop signal

        :return:
        """
        self.root.stop.set()

    def build(self):
        """
        gets called upon running the GUI Object, initialises the window as well as the the GUI Layout
        :return:
        """
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)

        layGUI = GUILayout()
        layGUI.initialize_std_out_redirection()

        return layGUI


def main():
    """
    Entry point for the programm
    Runs the GUI

    :return:
    """
    gui = RepositoryClassifierApp()
    gui.run()

if __name__ == "__main__":
    main()
