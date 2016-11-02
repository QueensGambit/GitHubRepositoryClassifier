# http://chase-seibert.github.io/blog/2016/07/22/pygithub-examples.html

from github3 import GitHub

gh = GitHub()

usrQG = gh.user('QueensGambit')
repo = gh.repository('QueensGambit', 'Barcode-App')
print('usrQG.msg:')
print(usrQG)
print('repos:')
print(repo)
lstEmojis = gh.emojis()
print('lstEmojis:')
print(lstEmojis)