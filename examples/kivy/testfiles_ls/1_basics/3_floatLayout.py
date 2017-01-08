"""
@file: 2_widget.py
Created on 05.01.2017 22:04
@project: GitHubRepositoryClassifier

@author: Lukas

Your description goes here
"""

import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout


class FloatingApp(App):
    def build(self):
        return FloatLayout()

flApp = FloatingApp()
flApp.run()
