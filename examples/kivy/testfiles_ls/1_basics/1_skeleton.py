"""
@file: 1_test.py
Created on 05.01.2017 21:58
@project: GitHubRepositoryClassifier

@author: Lukas

Your description goes here
"""

import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.uix.label import Label

class HelloKivyApp(App):

    def build(self):
        return Label()

helloKivy = HelloKivyApp()

helloKivy.run()