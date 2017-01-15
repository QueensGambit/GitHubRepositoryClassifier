"""
@file: rc_cli.py
Created on 15.01.2017 02:36
@project: GitHubRepositoryClassifier

@author: Anonym

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

import sys
import os


def printMenu():
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
    init()
    strInput = ""

    # initialize the repositoryClassifier
    repoClassifier = RepositoryClassifier(bUseStringFeatures=True)
    repoClassifier.loadModelFromFile(console=True)

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
    #Checks file exists and txt file
    if os.path.exists(strFileInput) & string_operation.validate_txtfile(strFileInput):
        file = open(strFileInput, 'r')

        strReadFileDirectory = os.path.dirname(strFileInput)
        strReadFileName = os.path.basename(strFileInput)

        print(strReadFileName + 'was read successfully')

        strFileClassified = strReadFileName + '_classified' + '.txt'
        iLabel = None

        writeClassifiedTxtFile(file, strReadFileDirectory, strFileClassified, repoClassifier)
    else:
        print("File could no be read. Make sure you have permission or entered correct File (txt)")


def writeClassifiedTxtFile(file, strReadFileDirectory, strFileClassified, repoClassifier):
    try:

        for line in file:
            strRepoUrl = line.rsplit('\n')
            iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL(str(strRepoUrl))

            with open(os.path.join(strReadFileDirectory, strFileClassified), 'wb') as temp_file:

                temp_file.write(line + ',' + iLabel + '\n')

        print(strFileClassified + ' was created and classified.')
    except OSError as err:
        print("Could not create file. Make sure you have permission in created Directory".format(err))

    #finally:
        #os.close(file)
        #os.close(temp_file)

def predictCategoryFromURL(repoClassifier, strUrl):

    iLabel, iLabelAlt, lstFinalPercentages, tmpRepo, lstNormedInputFeatures = repoClassifier.predictCategoryFromURL(
        strUrl)
    repoClassifier.printResult(tmpRepo, iLabel, iLabelAlt, bPrintWordHits=False)


if __name__ == "__main__":
    main()