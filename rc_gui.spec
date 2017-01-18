# -*- mode: python -*-
from kivy.deps import sdl2, glew #, gstreamer


# steps to build:
# (1) all needed dependecies to run the .py must be installed
# (2) -> setup tools must be downgraded to version 19.2
#	  pip install --ignore-installed setuptools==19.2
# (3) copy ".\GitHubRepositoryClassifier\gui_prototype\prototype\__pycache__\__init__.cpython-35.pyc" to
#	  ".\GitHubRepositoryClassifier\gui_prototype\prototype\__init__.pyc"
# (5) run "python -m compile all"
# (4) now you can build .exe via:
#     "python -m PyInstaller rc_gui.spec"


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
