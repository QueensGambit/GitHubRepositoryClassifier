"""
@file: gui_protoype.py
Created on 07.01.2017 23:06
@project: GitHubRepositoryClassifier

@author: Lukas

GUI Prototype using kivy
"""

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
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas     # don't worry, it works even though its red
from kivy.uix.popup import Popup
import clipboard
from sklearn import decomposition
import matplotlib.patches as mpatches

import sys, os
import matplotlib.pyplot as plt
# from colour import Color
from scipy.misc import imread

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
import time
import threading
from kivy.animation import Animation
from kivy.clock import Clock, mainthread
from kivy.factory import Factory

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image
from PIL import ImageOps
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
    b_run_loading = True                            # boolean to stop loading thread from running too late
    animation_loading = Animation()
    anim_bar = None
    str_stdOutPuffer = ""


class StdOut(object):
    def __init__(self, log_console, oldStdOut):
        # self.txtctrl = txtctrl
        self.log_console = log_console
        self.oldStdOut = oldStdOut

    def write(self, string):

        self.oldStdOut.write(string)
        StaticVars.str_stdOutPuffer += string


    def flush(self):
        pass


class Radar(object):

    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles)
        self.angles = np.arange(45, 45+360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                         for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 5), angle=angle, labels=label)
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

    def __init__(self, windowParent):
        """
        Called upon opening of the Settings Popup
        Override the active state of the API checkbox to display the current internal saved state

        :param windowParent: windows handle of the main frame where the console is located
        """
        super(SettingsPopup, self).__init__()
        self.checkbox_api_token.active = StaticVars.b_api_checkbox_state
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

    def update_console(self):

        self.log_console.text += StaticVars.str_stdOutPuffer
        StaticVars.str_stdOutPuffer = ""

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
            print('self.lstNormedInputFeatures: ', self.lstNormedInputFeatures[:4])

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

    def start_loading_animation(self, *args):
        """
        Creates the User-Feedback while loading, such as setting the label_error and showing the loading animation.

        :param args:
        :return:
        """

        if StaticVars.b_run_loading:

            print("Start loading animation")

            # self.button_classifier.disabled = True                      # disable button

            # Remove the button.
            self.layout_pie_chart.clear_widgets()
            # self.remove_widget(self.but_1)

            # Update a widget property.
            self.set_info("[INFO] Classification in progress")
            self.label_result.text = "Loading..."
            self.label_second_result.text = ""

            # Create and add a new widget.
            StaticVars.anim_bar = Factory.AnimWidget()
            self.layout_pie_chart.add_widget(StaticVars.anim_bar)

            # Animate the added widget.
            StaticVars.animation_loading = Animation(opacity=0.3, width=100, duration=0.6)
            StaticVars.animation_loading += Animation(opacity=1, width=400, duration=0.8)
            StaticVars.animation_loading.repeat = True
            StaticVars.animation_loading.start(StaticVars.anim_bar)
        else:
            print("didn't start loading animation")

    @mainthread
    def show_classification_result(self, iLabel, iLabelAlt, lstFinalPercentages, tmpRepo):
        """
        Creates the user output for the final result:
        The pie chart as well as the label in the top right corner

        :param iLabel: index of the Category Enum of the found result
        :param iLabelAlt: index of the Category Enum of the secondary result
        :param lstFinalPercentages: List of the percentages each Category yields
        :param tmpRepo: a Repository object
        :return:
        """

        self.layout_pie_chart.clear_widgets()

        if iLabel is not None:
            self.renderPieChart(lstFinalPercentages)

            lstFinalPercentages.sort()
            if lstFinalPercentages[5] > lstFinalPercentages[6] - .5:
                self.label_second_result.text = "Secondary Result: " + CategoryStr.lstStrCategories[iLabelAlt]

            self.label_result.text = 'Result: ' + CategoryStr.lstStrCategories[iLabel]

            self.set_info("[INFO] Classification complete")

            # Wordcloud
            strText = str(tmpRepo.getFilteredReadme(bApplyStemmer=True) + " " + tmpRepo.getFilteredRepoDescription(
                bApplyStemmer=True))

            if not strText.isspace():
                self.show_wordcloud(strText, iLabel)

            else:
                self.layout_diagram1.clear_widgets()
                self.layout_diagram1.add_widget(Label(text="The Repository doesn't contain any words"))

            # multidimensional

            self.plot_multi_dim()
            # self.plot_barchart(self.lstNormedInputFeatures)
            self.plot_net_diagram(tmpRepo)
        else:
            self.label_result.text = 'No Result'
            self.label_second_result = ""

        self.button_classifier.disabled = False                      # re-enable button
        StaticVars.b_run_loading = False
        StaticVars.animation_loading.cancel(StaticVars.anim_bar)
        self.update_console()

    def show_wordcloud(self, text, iLabel):
        """
        Creates the Wordcloud in the first Diagram Tab.

        :param text: The Text to create the word cloud from
        :param iLabel: index of the Category Enum of the found result
        :return:
        """

        # print('text: ', text)
        # img = (Image.open(self.strPath + "/media/icons/" + CategoryStr.lstStrIcons[iLabel])).split()[-1]
        # img = np.array((Image.open(self.strPath + "/media/icons/colored/" + 'code_v2_-8x_colored.png')))
        img = np.array((Image.open(self.strPath + "/media/icons/colored/" + CategoryStr.lstStrIcons[iLabel])))

        # imgColor = img.clone()
        # imgColor[:] = (24, 45, 23)
        # im1arrF = img.astype('float')
        # im2arrF = imgColor.astype('float')
        # additionF = (im1arrF + im2arrF) / 2
        # img = additionF.astype('uint8')

        # addition = (img + im2arr) / 2

        # the mask is inverted, so invert it again
        # img = ImageOps.invert(img)
        # img = img.resize((512, 512), Image.NONE)
        # imgMask = np.array(img)

        # create coloring from image
        # imgGrayVariance = Image.open(self.strPath + "/media/icons/" + "gray_variance.png") #imread(path.join(d, "alice_color.png"))
        img_colors = ImageColorGenerator(img)
        # wordcloud = WordCloud(background_color=(48, 48, 48), mask=imgMask).generate(text)
        wordcloud = WordCloud(background_color=(48, 48, 48), mask=img, color_func=img_colors).generate(text)
        self.layout_diagram1.clear_widgets()
        plt.figure(2)
        plt.imshow(wordcloud)
        # plt.imshow(wordcloud.recolor(color_func=img_colors))

        plt.axis("off")
        fig = plt.gcf()
        fig.patch.set_facecolor((48/255, 48/255, 48/255))
        self.layout_diagram1.add_widget(FigureCanvas(fig))

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
        webbrowser.open("http://google.com")

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
        print('pasted text:', clipboard.paste())

    def initialize(self):
        """
        Initializes the Main Layout (GUILayout), is called after the its creation in the RepositoryClassifierApp

        :return:
        """

        oldStdOut = sys.stdout
        # overload load the sys.strdout to a class-instance of StdOut
        sys.stdout = StdOut(self.log_console, oldStdOut)
        self.log_console.scroll_y = 0                             # makes the console scroll down automatically

        # initialize the repositoryClassifier
        self.repoClassifier = RepositoryClassifier(bUseStringFeatures=True)  #bUseStringFeatures=False
        self.clf, self.lstMeanValues, self.matIntegerTrainingData, self.lstTrainLabels, self.lstTrainData, self.normalizer, self.normalizerIntegerAttr = self.repoClassifier.loadModelFromFile()

        # self.strPath = os.path.dirname(__file__)
        self.strPath = str(Path())

        self.log_console.scroll_y = 0                                   # makes the console scroll down automatically


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

    def set_info(self, info):
        """
        put the info text as info text, text color to white

        :param info: the text to display
        :return:
        """
        self.label_info.color = 1, 1, 1, 1
        self.label_info.text = info

    def set_error(self, error):
        """
        put the info text as error text, text color to red

        :param error:
        :return:
        """
        self.label_info.color = 1, 0, 0, 1
        self.label_info.text = error

    def classify_button_pressed(self):
        """
        Gets called from the "Classify" button, starts the classification thread function

        :return:
        """

        url_in = "".join(self.textfield_input.text.split())               # read input and remove whitespaces
        self.textfield_input.text = url_in
        print("[INFO] Starting Process with \"" + url_in + "\"")        # print info to console
        valid = self.validate_url(url_in)                               # validate input and handle Errors

        if valid:
            self.button_classifier.disabled = True                      # disable button
            StaticVars.b_run_loading = True                             # enable loading screen
            self.start_classification_thread(self.label_info.text, url_in)

    def renderPieChart(self, lstFinalPercentages):
        """
        Creates the pie chart

        :param lstFinalPercentages: the percentages to use in the piechart
        :return:
        """
        print("[INFO] Rendering Piechart")

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
        fig = plt.figure(1, figsize=(10, 10), dpi=70)
        fig.clear()

        plt.pie(lstFinalPercentages, explode=explode, colors=CategoryStr.lstStrColors, labels=labels, autopct='%1.1f%%', shadow=True,
                  startangle=90)

        # plt.axis('equal')                                        # this was the actual cause of the resizing !!!
        #  -> this causes a warning; alternative us fig,set_tight_layout(True)
        # plt.tight_layout()                                       # http://matplotlib.org/users/tight_layout_guide.html

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

    def load_example(self, link):
        """
        write the example link into the textfield, gets called by the Examples in the Action Bar

        :param link: the link to the repository, without the github in front!
        :return:
        """
        self.textfield_input.text = "https://github.com/" + link

    def plot_multi_dim(self):
        """
        show multidimensional data in 2D plot
        :return:
        """

        clf = self.clf
        lstTrainLabels = self.lstTrainLabels
        data = self.matIntegerTrainingData

        if len(data) < 2:
            raise Exception('Lenght of array >= 2')

        normalizer = preprocessing.Normalizer()
        normalizer.fit(data)
        data = normalizer.fit_transform(data)

        if data.shape[1] > 2:
            pca = decomposition.PCA(n_components=2)
            pca.fit(data)
            data = pca.transform(data)

        n_clusters = 7
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kmeans.fit(data)
        h = .02

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(3)
        plt.cla()
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')
        # plt.plot(multidimarray[:, 0], multidimarray[:, 1], 'k.', markersize=2)

        lstColors = [None] * len(lstTrainLabels)

        for i, iLabel in enumerate(lstTrainLabels):
            lstColors[i] = CategoryStr.lstStrColors[iLabel]

        plt.scatter(data[:, 0], data[:, 1], cmap=plt.cm.Paired, color=lstColors)

        centroids = clf.centroids_
        centroids = normalizer.fit_transform(centroids)

        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color=CategoryStr.lstStrColors, zorder=10)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        # .show()

        lstPatches = [None] * len(CategoryStr.lstStrCategories)
        for i, strCategory in enumerate(CategoryStr.lstStrCategories):
            lstPatches[i] = mpatches.Patch(color=CategoryStr.lstStrColors[i], label=strCategory)

        plt.legend(handles=lstPatches)

        fig = plt.gcf()
        fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))
        self.layout_diagram3.clear_widgets()
        self.layout_diagram3.add_widget(FigureCanvas(fig))



    def getMedianValue(lstData):
        sortedlist = sorted(lstData, reverse=False)
        iSize = len(sortedlist)
        return sortedlist[iSize / 2]

    def plot_barchart(self, lsIntegerAttributes):
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

    def plot_net_diagram(self, repo):
        # http://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes
        import pylab as pl
        import math

        lsAttributes = self.lstNormedInputFeatures[0][:4]
        print(lsAttributes)
        print(repo.getIntegerFeatures())
        print(self.lstMeanValues)
        lstNormedMeanValues = self.normalizer.transform([1] * len(self.lstMeanValues))
        print(lstNormedMeanValues)

        print(math.log2(repo.getIntegerFeatures()[0]))
        print("normalizer params: " + str(self.normalizer.get_params()))


        fig = pl.figure(5, figsize=(0.1, 0.1))
        fig.clear()



        titles = ['Subscribers', 'Open Issues', 'DevTime', 'Size']

        labels = [
            [],
            [],
            [],
            []
        ]

        radar = Radar(fig, titles, labels)
        radar.plot(lstNormedMeanValues[0] * 5, "-", lw=2, color="purple", alpha=0.4, label="Average")
        radar.plot(lsAttributes * 5, "-", lw=2, color="r", alpha=0.4, label="This Repo")
        radar.ax.legend(loc=(1, .6))

        fig = pl.gcf()
        fig.patch.set_facecolor((48 / 255, 48 / 255, 48 / 255))
        self.layout_diagram2.clear_widgets()
        self.layout_diagram2.add_widget(FigureCanvas(fig))


class FileSaverPopup(Popup):
    """
    The Popup to save the console output to a log file. called by save_log() in the GUILayout class.
    """
    label_save = ObjectProperty()           # the label to output potential error messages in the File Saver Popup
    def __init__(self, windowParent):
        super(FileSaverPopup, self).__init__()
        self.windowParent = windowParent

    def save_file(self, path, filename):
        """
        Called by the "save" button in the Popup.
        Saves the file with the input filename to the path. Handles Permission Errors.

        :param path: The current folder path to save the file into, ends with "/"
        :param filename: The filename to save the file as
        :return:
        """
        try:
            with open(os.path.join(path, filename), 'w') as stream:
                stream.write(self.log_text)
                stream.close()
            self.dismiss()
            print("[INFO] Logfile saved to: " + path + "\\" + filename)
            # print("[INFO] Logfile saved to: " + os.path.join(path, filename) + "\\" + filename)
        except PermissionError as err:
            print("[ERROR] Logfile couldn't be saved. Permission denied. Path: " + path + "\nError: " + str(err))
            self.label_save.text = "[ERROR] Couldn't save. Permission denied."
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
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        self.root.stop.set()

    def build(self):
        """
        gets called upon running the GUI Object, initialises the window as well as the the GUI Layout
        :return:
        """
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        # Window.size = (1200, 800)

        layGUI = GUILayout()
        layGUI.initialize()

        return layGUI


gui = RepositoryClassifierApp()
gui.run()

# TODO: PRIORITY - DESCRIPTION
# TODO: VERY HIGH - Build Windows-Excecutable
# TODO: VERY HIGH - Write Docu
# TODO: HIGH - Build single console excecutable
# TODO: HIGH - Sometimes the program crashes maybe because of thread scheduling
# TODO: HIGH - Sometimes the plots are drawn in the wrong windows/layouts (wordcloud and pie-chart)
# TODO: HIGH - Add missing plots (bar-chart as well as visual-2D-Map)
# TODO: MEDIUM - Build Sphinx-Docu
# TODO: MEDIUM - Beautifully draw the word clouds in different colors
# TODO: LOW - Drob-Down-List with nice examples which work well ;)
# TODO: LOW - Add better animation while waiting


