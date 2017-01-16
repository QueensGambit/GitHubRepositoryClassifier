"""
@file: rc_cli.py
Created on 15.01.2017 02:36
@project: GitHubRepositoryClassifier

@author: QueensGambit

Your description goes here...
"""


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

# pip install prettytable
# ...
# Successfully installed prettytable-0.7.2


from prettytable import PrettyTable
from prototype.repository_classifier import RepositoryClassifier
from prototype.utility_funcs import string_operation
from prototype.definitions.categories import Category

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
    t = PrettyTable(['Action', '    Shortcute   '])
    t.add_row(['Show Menu', '- m -'])
    t.add_row(['     Predict repositories form txt-file     ', '- i -'])
    t.add_row(['Input URL', '- u -'])
    t.add_row(['Show Info', '- f -'])
    t.add_row(['Train Model', '- t -'])
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

def main():
    """
    predicting repositories headless
    :return:
    """
    init()
    strInput = ""

    # initialize the repositoryClassifier
    repoClassifier = RepositoryClassifier(bUseStringFeatures=True)
    repoClassifier.loadModelFromFile()

    while strInput != 'q':
        strInput = input()

        if strInput == 'm':
            printMenu()

        elif strInput == 'i':
            print("Enter path of file")
            strFileInput = input()
            predictFromFile(repoClassifier, strFileInput)

        elif strInput == 'u':
            print("Enter the URL to a Repository.")
            strUrlInput = input()
            predictCategoryFromURL(repoClassifier, strUrlInput)

        elif strInput == 'f':
            print('Show Info')

        elif strInput == 'h':
            print('help...')

        #striagt url
        elif len(strInput) > 1 and string_operation.validate_url(strInput):
            predictCategoryFromURL(repoClassifier, strInput)

        #straigt file
        elif len(strInput) > 1 and string_operation.validate_txtfile(strInput):
            predictFromFile(repoClassifier, strInput)



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

        strFileClassified = strReadFileName + '_classified' + '.txt'

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

            classifiedFile.write(strRepoUrl + ',' + str(Category(iLabel)) + '\n')

        print(strFileClassified + ' was created and classified.')
    except OSError as err:
        print("Could not create file. Make sure you have permission in created Directory".format(err))

    finally:
        file.close()
        classifiedFile.close()

def predictCategoryFromURL(repoClassifier, strUrl):
    """
    Predicts category of repository
    :param repoClassifier:
    :param strUrl:
    :return:
    """
    iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL(
        strUrl)
    repoClassifier.printResult(tmpRepo, iLabel, iLabelAlt, bPrintWordHits=False)


if __name__ == "__main__":
    main()