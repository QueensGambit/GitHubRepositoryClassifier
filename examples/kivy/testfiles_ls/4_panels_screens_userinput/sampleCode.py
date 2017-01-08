"""
@file: sampleCode.py
Created on 07.01.2017 22:07
@project: GitHubRepositoryClassifier

@author: Lukas

Your description goes here
"""

import kivy

kivy.require("1.9.0")

from kivy.app import  App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.popup import Popup


class CustomPopup(Popup):
    pass


class SampleBoxLayout(BoxLayout):
    checkbox_is_active = ObjectProperty(False)

    def checkbox_18_clicked(self, instance, value):
        if value is True:
            print("Checkbox Checked")
        else:
            print("Checkbox Unchecked")

    blue = ObjectProperty(True)
    red = ObjectProperty(False)
    green = ObjectProperty(False)

    def switch_on(self, instance, value):
        if value is True:
            print("Switch on")
        else:
            print("Switch off")

    def open_popup(self):
        the_popup = CustomPopup()
        the_popup.open()

    def spinner_clicked(self, value):
        print("Spinner Value " + value)


class SampleApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        return SampleBoxLayout()

sample_app = SampleApp()
sample_app.run()
