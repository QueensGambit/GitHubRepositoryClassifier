# GitHubRepositoryClassifier

1. [Introduction](#introduction)
2. [Installation](#installation)

## Introduction
### subject of the [informatiCup 2016/2017](https://github.com/InformatiCup/InformatiCup2017/)


This program will take an URL of a github repository as the input and will try to assign it to one of seven Categories.

The Categories are:
* **DEV** - a repository primarily used for development of a tool, component, application, app or API
* **HW** - a repository primarily used for homework, assignments and other course-related work and code.
* **EDU** - a repository primarily used to host tutorials, lectures, educational information and code related to teaching
* **DOCS** - a repository primarily used for tracking and storage of non-educational documents
* **WEB** - a repository primarily used to host static personal webpages or blogs
* **DATA** - a repository primarily used to store data sets
* **OTHER** - this category is used only if there is no strong correlation to any other repository category


## Installation

### Windows (Executable)

We built an executable for Windows-x64 systems which should work on most systems.
No Python installation and no external library, except Open-GL 2.0 (which should be installed by default on your system) is required to run the executable.
Should you encounter any problem you can follow the guide to run the [python-version](#python-version).

*(e. g. the Python version support Open-GL Versions < 2.0)*

[You can download the zip-file from here](https://www.dropbox.com/s/p6dvmt5xtdazjaz/GitHubClassifier.zip?dl=0), unzip it to an arbitrary location and execute one of the batch files.

You can launch the GUI via **rc_gui.bat** and the CLI via **rc_cli.bat**.

![RC_GUI WINDOWS](https://raw.githubusercontent.com/QueensGambit/GithubRepositoryClassifier/master/gui_prototype/media/readme/rc_gui_windows.png "rc")

![RC_CLI WINDOWS](https://raw.githubusercontent.com/QueensGambit/GithubRepositoryClassifier/master/gui_prototype/media/readme/rc_cli_windows.png)

The rc_cli will provide a way for you to use the classifier with the commandline, including a console-menu.
This even accepts arguments directly so you can automate the process if you wish to do so.
You can even give it a .txt file with many repositories to classify and it will create a new file with the results for you.

*Note: If you have troble executing the GUI, try starting the script from the commandline first. After that you should be able to use the batch or exe file without any problems.*

### Python-Version

*[optional]* Create a virtual environment in a directory of your choice and activate it.
```
pip3 install virtualenv
virtualenv --no-site-packages env
```

#### Windows (Python-Version)

*The Python Version has been tested with Python 3.5.1*

*[optional]* Activate the virtual environment:
```
env/Scripts/activate.bat
````

It's possible to install the Anaconda Distribution which provides most packages from the start:

Install all needed requirements:
```
pip install -r requirements.txt
```

#### Linux (Ubuntu)
*[optional]* Activate the virtual environment:
```
source my_project/bin/activate
```

type:
```
pip install -r requirements.txt
```

![RC_GUI UBUNTU](https://raw.githubusercontent.com/QueensGambit/GithubRepositoryClassifier/master/gui_prototype/media/readme/rc_gui_ubuntu.png)

The app has been tested on Ubuntu 14.04 LTS with Python 3.4.1.
## Documentation

Please see the [wiki](https://github.com/QueensGambit/GitHubRepositoryClassifier/wiki/Documentation)

# Info

<!--![HFU](https://cloud.githubusercontent.com/assets/7681159/22186901/aaf6289c-e0fd-11e6-8b87-590aa5606871.png)-->
![RC_GUI UBUNTU](https://raw.githubusercontent.com/QueensGambit/GithubRepositoryClassifier/master/gui_prototype/media/readme/hfu-logo.png)
