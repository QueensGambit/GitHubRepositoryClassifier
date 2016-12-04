from kivy.app import App

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label


class TutorialApp(App):
    def build(self):
        f = FloatLayout()
        s = Scatter()
        l = Label(text="Hello!", font_size=150)
        f.add_widget(s)
        f.add_widget(l)
        return f

if __name__ == "__main__":
    TutorialApp().run()