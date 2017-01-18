# -*- mode: python -*-
from kivy.deps import sdl2, glew #, gstreamer


# steps to build:
# (0) all needed dependecies to run the .py must be installed
# (1) in addition you must install:
#     pip install --upgrade pyinstaller
#     (when this doesn't work: pip install --ignore-installed --upgrade pyinstaller)
#     install sdl2 by downloading and installing the .whl from: https://pypi.python.org/pypi/kivy.deps.sdl2
#     pip install kivy.deps.sdl2-0.1.17-cp35-cp35m-win_amd64.whl
#     install glew by downloading and installing the .whl from: https://pypi.python.org/pypi/kivy.deps.glew
#     pip install kivy.deps.glew-0.1.9-cp35-cp35m-win_amd64.whl
# (2) -> setup tools must be downgraded to version 19.2
#     pip uninstall setuptools
#	  pip install --ignore-installed setuptools==19.2
# (3) copy ".\GitHubRepositoryClassifier\gui_prototype\prototype\__pycache__\__init__.cpython-35.pyc" to
#	  ".\GitHubRepositoryClassifier\gui_prototype\prototype\__init__.pyc"
# (5) run "python -m compileall ." in directory ./GithubClassifier/
# (4) now you can build .exe via:
#     python -m PyInstaller rc_gui.spec


block_cipher = None

a = Analysis(['.\\gui_prototype\\rc_gui.py'],
             pathex=['G:\\Programming\\Projects\\Python\\PyInstallerTest\\GithubClassifier'],
             binaries=None,
             datas=None,
			 # 'wordcloud', 'wordcloud.WordCloud', 'wordcloud.ImageColorGenerator' is not recognized
             hiddenimports=['kivy.garden', 'matplotlib.pyplot', 'kivy.garden.matplotlib', 'cython', 'sklearn', 'sklearn.neighbors.typedefs', 'os.path.expanduser', 'win32timezone'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='rc_gui',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='.\\gui_prototype\\media\\icons\\rc_icon.ico')
		  
coll = COLLECT(exe, Tree('G:\\Programming\\Projects\\Python\\PyInstallerTest\\GithubClassifier\\gui_prototype\\'),
               a.binaries,
               a.zipfiles,
               a.datas,
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               name='rc_gui')
