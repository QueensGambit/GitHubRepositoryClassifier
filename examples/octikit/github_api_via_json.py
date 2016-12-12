# -------------------------------------------------------------------------------------------
#http://stackoverflow.com/questions/10625190/most-suitable-python-library-for-github-api-v3

import requests
import json
import os
from os import path



# now the json-files are exported after reading, in order to avoid the api-request limit
# and to facilitate the reproducing of the learning phase
# only make a json-request if the json-file hasn't been requested already
d = path.dirname(__file__)
strPathDjangoJSON = d + '/json/django_django.json'

# define the json_data for the django variable
repoDjango = None

if os.path.isfile(strPathDjangoJSON):
    # file already exists
    # read from it
    with open(strPathDjangoJSON) as json_data:
        repoDjango = json.load(json_data)
    print('json-file was read')
else:
    print('request json-file from Github...')
    requests.add
    r = requests.get('https://api.github.com/repos/django/django')

    if r.ok:
        repoDjango = json.loads(r.text or r.content)

        # export to json-file
        with open(strPathDjangoJSON, 'w') as outfile:
            json.dump(repoDjango, outfile)
            print('json-data was exported to: ', strPathDjangoJSON)

# print out the results
print('~~~~~~~~~ INFOS ABOUT github.com/repos/django/django ~~~~~~~~~')
print(" id: " + str(repoDjango['id']))
print("repository created: " + repoDjango['created_at'])
print("owner_login: " + repoDjango['owner']['login'])
print("url: " + repoDjango['url'])

# -------------------------------------------------------------------------------------------

'''
rEmoji = requests.get('https://api.github.com/repos/WebpageFX/emoji-cheat-sheet.com')

if rEmoji.ok:
    repoEmoji = json.loads(rEmoji.text or rEmoji.content)
    print('~~~~~~~~~ INFOS ABOUT github.com/repos/WebpageFX/emoji-cheat-sheet.com ~~~~~~~~~')
    print('description: ' + repoEmoji['description'])
    print('size: ' + str(repoEmoji['size']))
    print('watcher: ' + str(repoEmoji['watchers']))
'''

