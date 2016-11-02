# -------------------------------------------------------------------------------------------
#http://stackoverflow.com/questions/10625190/most-suitable-python-library-for-github-api-v3

import requests
import json
r = requests.get('https://api.github.com/repos/django/django')

if r.ok:
    repoDjango = json.loads(r.text or r.content)
    print('~~~~~~~~~ INFOS ABOUT github.com/repos/django/django ~~~~~~~~~')
    print(" id: " + str(repoDjango['id']))
    print("repository created: " + repoDjango['created_at'])
    print("owner_login: " + repoDjango['owner']['login'])
    print("url: " + repoDjango['url'])

# -------------------------------------------------------------------------------------------


rEmoji = requests.get('https://api.github.com/repos/WebpageFX/emoji-cheat-sheet.com')

if rEmoji.ok:
    repoEmoji = json.loads(rEmoji.text or rEmoji.content)
    print('~~~~~~~~~ INFOS ABOUT github.com/repos/WebpageFX/emoji-cheat-sheet.com ~~~~~~~~~')
    print('description: ' + repoEmoji['description'])
    print('size: ' + str(repoEmoji['size']))
    print('watcher: ' + str(repoEmoji['watchers']))
