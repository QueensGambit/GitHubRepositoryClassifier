"""
@file: switch_and_builder.py
Created on 07.01.2017 22:45
@project: GitHubRepositoryClassifier

@author: Lukas

Your description goes here
"""

import kivy
kivy.require("1.9.0")

from kivy.app import  App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager


Builder.load_string("""
<ScreenOne>:
    BoxLayout:
        orientation: "vertical"
        size_hint_y: .5
        Label:
            text: "Screen 1"
        Button:
            text: "Go to Screen 2"
            on_press:
                root.manager.transition.direction = "left"
                root.manager.transition.duration = 2
                root.manager.current = "screen_two"
<ScreenTwo>:
    BoxLayout:
        Label:
            text: "Screen 2"
        Button:
            text: "Go to Screen 3"
            on_press:
                root.manager.transition.direction = "left"
                root.manager.transition.duration = 2
                root.manager.current = "screen_three"
<ScreenThree>:
    BoxLayout:
        Button:
            text: "Go to Screen 1"
            on_press:
                root.manager.transition.direction = "left"
                root.manager.transition.duration = 2
                root.manager.current = "screen_one"

""")

class ScreenOne(Screen):
    pass

class ScreenTwo(Screen):
    pass

class ScreenThree(Screen):
    pass

screen_manager = ScreenManager()

screen_manager.add_widget(ScreenOne(name="screen_one"))
screen_manager.add_widget(ScreenTwo(name="screen_two"))
screen_manager.add_widget(ScreenThree(name="screen_three"))

class ScreenSwapperApp(App):
    def build(self):
        return screen_manager

screen_app = ScreenSwapperApp()
screen_app.run()
