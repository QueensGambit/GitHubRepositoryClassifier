"""
@file: rc_cli.py
Created on 15.01.2017 02:36
@project: GitHubRepositoryClassifier

@author: NexusHero

Your description goes here...
"""


# This block is needed to use the right dll-file for building
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import ctypes
import os
import sys

# this was needed to build the windows executable
if getattr(sys, 'frozen', False):
  # Override dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW('G:/Program Files/Anaconda3/Library/bin')
  # Init code to load external dll
  ctypes.CDLL('mkl_avx2.dll')
  ctypes.CDLL('mkl_def.dll')
  ctypes.CDLL('mkl_vml_avx2.dll')
  ctypes.CDLL('mkl_vml_def.dll')

  # Restore dll search path.
  ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


strStopper1 = "=" * 79
strStopper2 = "-" * 79

# this ascii-art was created with:
# https://gist.github.com/cdiener/10567484

strLogoGithub = "\
        ;;;; :;;;;.;;;;\n\
        ;;;;;;;;;;;;;;;\n\
        ;;;;;;;;;;;;;;;;\n\
       ;;;;;;;;;;;;;;;;;;          iiii,\n\
       ;;;            ;;;        iiiiiii  iii  iii   ii    ii          iii\n\
       ;;             :;        iii            iii   ii    ii          iii\n\
        ;;            ;;        ii   ;;;; iii iiiii  iiiiiiii iii  iii iiiiiii;\n\
          ;;;      :;;          ii   iiii iii  iii   iiiiiiii iii  iii iii  iii\n\
             ;;;;;;             iii   iii iii  iii   ii    ii iii  iii iii  iii\n\
         ;;;;;;;;;;.             iiiiiiii iii  iiiir ii    ii .iiiiiii iiiiiiii\n\
            ;;;;;;;.               iiii   ;;;   ,ii  ii    ii   iii ii ii iii \n\
            ;;;;;;;\n\
              ;;;;"


strLogoRC = "\
         iiiiiiii;\n\
     iiiiiiiiiiiiiiiiii\n\
      iiiiiiiiiiiiiii\n\
     iiiiiiiiiiiiiiiii\n\
 .   iiiiiiiiiiiiiiiii   .\n\
  ;; i ;iiiiiiiiiii; i ;;   ;;;;;;\n\
  ;; i ,iiii               ;;;;;;             ;;;;;;;;;;;   ;;;;;;;;;;;\n\
  ;; iiiiii ;;;;;;;;;;;;;;;;;;;;;;;        :;;;;;  ;;;;;  ;;;;;  ;;;;;;\n\
  ;; iiiii;;;;;;;;;;;;;;;;;;;;;;;;.       ;;;;;;  ;;;;; ;;;;;;  ;;;;;\n\
  ;; iii ;;;;;;;;;;;;;;;;;;;;;;;         ;;;;;         ;;;;;\n\
  ;; ii ;;;;;;;  ; :  :  ;;;;;;         ;;;;;         ;;;;;\n\
  ;; i ;;;;;;;  ;;,    ;;;;;;;.       ;;;;;;        ;;;;;;  ;;;;;\n\
  ;; ;;;;;;;;;;;;;;;;;;;;;;;         ;;;;;         ;;;;;;  ;;;;;\n\
  ; ;;;;;;;;;;;;;;;;;;;;;;;          ;;            ;;;;;;;;;;\n\
   ;;;;;;;;;;;;;;;;;;;;;;;\n\
"

help = "This application classifies github repositories. There are 7 Categories: DEV, HW, EDU, DOCS, WEB, DATA, OTHER.\
 The classification method is based on the Nearest Centroid algorithm of the scikit learn libary to navigate through the application"\
       "use the given menu."

info = "This application is developed by Björn Beha, Johannes Chzech, Lukas Scheuerle and Suhay Sevinc. "
# pip install prettytable
# ...
# Successfully installed prettytable-0.7.2


from prettytable import PrettyTable
from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs import string_operation
from prototype.definitions.categories import CategoryStr
from prototype.utility_funcs.io_agent import InputOutputAgent
import sys
import os


def printMenu():
    """
    prints menu guide for headless
    :return:
    """
    # tWelc = PrettyTable(['Welcome to the CLI-of the repository classifier'])
    print('Welcome to the CLI of the repository classifier')
    print(strStopper1)
    t = PrettyTable(['Action', '    Shortcut    '])
    t.add_row(['Show Menu', '- m -'])
    t.add_row(['     Predict repositories form txt-file     ', '- i -'])
    t.add_row(['Input URL', '- u -'])
    t.add_row(['Show Info', '- f -'])
    t.add_row(['Train Model', '- t -'])
    t.add_row(['set GitHub-Token', '- g -'])
    t.add_row(['Help', '- h -'])
    t.add_row(['Quit', '- q -'])
    print(t)
    print('')

def init():
    print(strStopper2)
    print(strLogoGithub)
    print()
    print(strStopper2)
    print()
    print(strLogoRC)
    print(strStopper2)
    print()
    printMenu()

# initialize the repositoryClassifier
repoClassifier = RepositoryClassifier(bUseStringFeatures=True)
repoClassifier.loadModelFromFile()

def main():
    """
    predicting repositories headless
    :return:
    """

    # change the dirctory to the current file
    # this is needed to run the script form every location on the system
    # http://stackoverflow.com/questions/1432924/python-change-the-scripts-working-directory-to-the-scripts-own-directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    if len(sys.argv) != 1:
        strParameter = sys.argv[1]

        if string_operation.validate_url(strParameter):
            repoClassifier.predictCategoryFromURL(strParameter)


    else:
        init()
        strInput = ""

        token = False
        while strInput != 'q':
            strInput = input()

            strInput = "".join(strInput.split())

            if strInput == 'm':
                printMenu()

            elif strInput == 'i':
                print("Enter path of file")
                strFileInput = input()
                predictFromFile(repoClassifier, strFileInput)

            elif strInput == 'u':
                print("Enter the URL to a Repository.")

                strUrlInput = input()
                url = "".join(strUrlInput.split())
                try:
                    if len(url) > 1 and string_operation.validate_url(url):
                         repoClassifier.predictCategoryFromURL(url)
                    else:
                        print("Make sure that you entered a correct url")
                except Exception as ex:
                    print("Exception has occured.")
                    if hasattr(ex, 'message'):
                        print(ex.message)
                    else:
                        print(ex)

            elif strInput == 'g':
                token = not token
                InputOutputAgent.setWithToken(token)

            elif strInput == 'f':
                print(info)

            elif strInput == 't':
                print("1. load external train data set.")
                print("2. load standard train data set.")

                strOption = input()
                try:
                    if strOption == "1":
                        print("hint: You will override the given train model. Are you sure you want to do this?  <y>")

                        strAwnser = input()

                        if strAwnser == "y" or strAwnser == "yes":
                            print("Enter a valid path of train data (.csv)")
                            strTrain = input()
                            lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData(strTrain, True)
                            repoClassifier.trainModel(lstTrainData, lstTrainLabels)
                            repoClassifier.exportModelToFile()
                            print("Model is trained and exported")
                        else:
                            print("User refused to learn new model")

                    elif strOption == "2":
                        print("Standard model will be loaded")
                        lstTrainData, lstTrainLabels = repoClassifier.loadTrainingData(
                            '/data/csv/additional_data_sets_cleaned.csv')
                        repoClassifier.trainModel(lstTrainData, lstTrainLabels)
                        repoClassifier.exportModelToFile()
                        print("standard model is loaded")
                    else:
                        print("User refused to learn new model")
                except:
                    print("Error occured while training. Pls try again!")

            elif strInput == 'h':
                print(help)

            #striagt url
            elif len(strInput) > 1 and string_operation.validate_url(strInput):
                repoClassifier.predictCategoryFromURL(strInput)

            #straigt file
            elif len(strInput) > 1 and string_operation.validate_txtfile(strInput):
                predictFromFile(repoClassifier, strInput)

            elif strInput != "q":
                print("no valid input! Use given menu")


def predictFromFile(repoClassifier, strFileInput):
    """
    Classifies a Repository list in txt file and creates a new file which contains the classified repositories
    :param repoClassifier:
    :param strFileInput:
    :return:
    """
    #Checks file exists and txt file
    if os.path.exists(strFileInput) & string_operation.validate_txtfile(strFileInput):
        file = open(strFileInput, 'r')

        strReadFileDirectory = os.path.dirname(strFileInput)
        strReadFileName = os.path.basename(strFileInput)

        print(strReadFileName + 'was read successfully')
        strFileClassified = "classified_" + strReadFileName

        writeClassifiedTxtFile(file, strReadFileDirectory, strFileClassified, repoClassifier)
    else:
        print("File could no be read. Make sure you have permission or entered correct File (txt)")


def writeClassifiedTxtFile(file, strReadFileDirectory, strFileClassified, repoClassifier):
    """
    creates  txt file which contains classified repositories.
    :param file:
    :param strReadFileDirectory:
    :param strFileClassified:
    :param repoClassifier:
    :return:
    """
    classifiedFile = None

    try:

        classifiedFile = open(strReadFileDirectory + '/' + strFileClassified, 'w')  # Trying to create a new file or open one

        for line in file:
            strRepoUrl = line.strip(os.linesep)
            iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL(strRepoUrl)

            classifiedFile.write(strRepoUrl + ' ' + CategoryStr.lstStrCategories[iLabel] + '\n')

        print(strFileClassified + ' was created and classified.')
    except OSError as err:
        print("Could not create file. Make sure you have permission in created Directory".format(err))

    finally:
        file.close()
        classifiedFile.close()



if __name__ == "__main__":
    main()
