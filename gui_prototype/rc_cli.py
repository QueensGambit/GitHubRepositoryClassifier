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
import sys
print(sys.argv)

def printMenu():
    # tWelc = PrettyTable(['Welcome to the CLI-of the repository classifier'])
    print('Welcome to the CLI of the repository classifier')
    print(strStopper1)
    t = PrettyTable(['Action', '    Shortcute   '])
    t.add_row(['Show Menu', '- m _'])
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
    while strInput != 'q':
        strInput = input()

        if strInput == 'h':
            print('help')
        elif strInput == 'm':
            printMenu()



if __name__ == "__main__":
    main()