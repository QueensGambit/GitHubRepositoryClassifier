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
from kivy.uix.stacklayout import StackLayout


class StackApp(App):
    def build(self):
        return StackLayout()

stackApp = StackApp()
stackApp.run()
