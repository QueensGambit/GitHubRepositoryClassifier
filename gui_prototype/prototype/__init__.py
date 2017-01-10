# import os, sys
# add the current directory to the system path in order to find the modules in relative path
# sys.path.insert(0, __path__)
# sys.path.append(os.path.abspath(".."))

from .utility_funcs import *
# from definitions import *

# from github_repo import GithubRepo
# from utility_funcs.io_agent import InputOutputAgent

from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())