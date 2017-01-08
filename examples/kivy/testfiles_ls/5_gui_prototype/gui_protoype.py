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

kivy.require("1.9.0")


class ScrollableLabel(ScrollView):
    text = StringProperty('')


class GUILayout(BoxLayout):
    url_input = ObjectProperty()
    classifier_button = ObjectProperty()
    error_label = ObjectProperty()
    log_console = ObjectProperty()

    def classify_button_pressed(self):
        print("Button pressed!")
        self.classifier_button.text = "pressed!"
        self.classifier_button.disabled = True
        self.error_label.text = "ERROR: Prototype not hooked up yet!"
        self.error_label.color = 1, 0, 0, 1
        self.log_console.text = ""
        self.log_console.scroll_y = 0                                   # makes the console scroll down automatically
        for i in range(0, 50):
            self.log_console.text += ("Button pressed, " + str(i) + "\n")



class GUIApp(App):
    def build(self):
        Window.clearcolor = ((41/255), (105/255), (176/255), 1)
        Window.size = (1200, 800)
        return GUILayout()

gui = GUIApp()
gui.run()



