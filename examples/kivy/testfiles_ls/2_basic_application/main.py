"""
@file: main.py
Created on 05.01.2017 22:52
@project: GitHubRepositoryClassifier

@author: Lukas

Your description goes here
"""

import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.uix.gridlayout import GridLayout


class CalcGridLayout(GridLayout):

    def calculate(self, calculation):
        if calculation:
            try:
                self.display.text = str(eval(calculation))
            except Exception:
                self.display.text = "ERROR"


class CalculatorApp(App):
    def build(self):
        return CalcGridLayout()

calcApp = CalculatorApp()
calcApp.run()
