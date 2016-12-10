from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.config import Config

'''class Ex11(RelativeLayout):
    pass'''


class StartScreen(GridLayout):
    def __init__(self, **kwargs):
        super(StartScreen, self).__init__(**kwargs)
        self.cols = 1
        # self.size =
        self.add_widget(Label(text='[b]Link to Repository[/b]', markup=True, size=(500, 30), size_hint=(1, 1)))
        self.link = TextInput(text='http://', multiline=False, size=(500, 30), size_hint=(1, 1))
        self.add_widget(self.link)
        self.add_widget(Button(text='Start', size=(100, 25), size_hint=(None, None)))


class RepositoryClassifierApp(App):
    Window.clearcolor = (52/255, 73/255, 94/255, 1)
    Window.size = (500, 100)

    def build(self):
        return StartScreen()

# 34495e
if __name__ == '__main__':
    RepositoryClassifierApp().run()
