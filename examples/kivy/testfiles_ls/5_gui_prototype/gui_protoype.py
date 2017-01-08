"""
@file: gui_protoype.py
Created on 07.01.2017 23:06
@project: GitHubRepositoryClassifier

@author: Lukas

GUI Prototype using kivy
"""

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

import matplotlib.pyplot as plt

kivy.require("1.9.0")


class ScrollableLabel(ScrollView):
    text = StringProperty('')


class GUILayout(BoxLayout):

    # define the ObjectProperties to communicate with the .kv file
    url_input = ObjectProperty()                    # user input line
    classifier_button = ObjectProperty()            # the button above the user input. should start the process
    error_label = ObjectProperty()                  # the label underneath the user input. short descriptions go here
    log_console = ObjectProperty()                  # console to output logs. just add text to it.
    result_label = ObjectProperty()                 # the big Result Label in the result corner
    pie_chart = ObjectProperty()                    # the Layout for the piechart in the result corner
    diagram1 = ObjectProperty()                     # the three TabbedPanelItems to put a diagram, expand if needed
    diagram2 = ObjectProperty()                     # ↑
    diagram3 = ObjectProperty()                     # ↑
    dialabel1 = ObjectProperty()

    def classify_button_pressed(self):
        print("Button pressed!")
        self.classifier_button.text = "pressed!"                        # button demo
        self.classifier_button.disabled = True
        self.error_label.text = "ERROR: Prototype not hooked up yet!"   # error label demo
        self.error_label.color = 1, 0, 0, 1
        self.log_console.text = ""                                      # clear console
        self.log_console.scroll_y = 0                                   # makes the console scroll down automatically
        for i in range(0, 50):                                          # demonstrate console
            self.log_console.text += ("Button pressed, " + str(i) + "\n")

        button = Button(text='Testbutton')                              # demonstration of clearing an area and adding a
        self.pie_chart.clear_widgets()                                  # widget to it
        self.pie_chart.add_widget(button)


        # ADDING A PIE CHART!
        # The slices will be ordered and plotted counter-clockwise.
        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        sizes = [15, 30, 45, 10]
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        explode = (0.1, 0, 0, 0)  # only "explode" the 1st slice (i.e. 'Dogs')
        plt.figure(1, figsize=(300, 300))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        fig = plt.gcf()
        fig.patch.set_facecolor('none')
        print(plt.colors())

        #plt.show()

        self.pie_chart.clear_widgets()
        self.pie_chart.add_widget(FigureCanvas(fig))       # das hier sollte laufen

    def save_log(self):
        print("save log")




class GUIApp(App):
    def build(self):
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        Window.size = (1200, 800)
        return GUILayout()

gui = GUIApp()
gui.run()



