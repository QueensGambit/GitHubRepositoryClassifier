from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.slider import Slider

class TutorialApp(App):
    def build(self):
        return Button(text="Hello!",
                      background_color=(108/255, 165/255, 69/255, 1.0),
                      font_size=150)


if __name__ == "__main__":
    TutorialApp().run()