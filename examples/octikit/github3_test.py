# http://chase-seibert.github.io/blog/2016/07/22/pygithub-examples.html

# for installtion use:
# pip install github3.py
from github3 import GitHub
from github3 import login
from github3.models import GitHubCore
import os
import json
import base64

def callAnonymous():
    # anonymous usage
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


def callByToken():

    gh = login(token=str(os.environ['TokenGithubAPI']))
    repoDjango = gh.repository('Django', 'Django')
    jsDjango = repoDjango.as_dict() #.as_json()
    code64README = repoDjango.readme().content #() #.get('content') # u get the content base64-coded
    strREADME = base64.b64decode(code64README)
    print('~~~~~~~ API-DJANGO-JASON ~~~~~~~')
    print(jsDjango)
    print('name:', jsDjango['name'])
    print('code64README:', code64README)
    print('strREADME:', strREADME)

#callAnonymous()
callByToken()

