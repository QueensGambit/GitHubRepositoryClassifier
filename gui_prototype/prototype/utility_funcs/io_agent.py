"""
@file: io_agent.py
Created on 11.12.2016 19:59
@project: GitHubRepositoryClassifier

@author: QueensGambit

The InputOutputAgent loads data (json-Data, README...) from a given repository which
 is defined by strUser and strName. If the needed data has already been requested before,
 then is loaded from a file. Otherwise a new connection is created.
 By default the autorization of the connection is done with an API-Token
"""
from _ast import In

from clyent import json_help
from docutils.io import Input

import json
import requests
import base64

# for installtion use:
# pip install github3.py
from github3 import GitHub
from github3 import login
import os


class InputOutputAgent:

    __gh = None
    __bWithToken = False
    __bWithTokenUpdated = False

    def __init__(self, strUser, strName):
        """
        Initializes the member variables

        :param strUser: username of the owner of the repository
        :param strName: name of the repository
        :param bWithToken:  checks if a github-token shall be used for a higher api-limit

        """

        self.strUser = strUser
        self.strName = strName

        self.strAPIUrl = "https://api.github.com/repos/" + strUser + "/" + strName
        self.lstReadmePath = {"https://raw.githubusercontent.com/" + strUser + "/" + strName + "/master/README.md",
                              "https://raw.githubusercontent.com/" + strUser + "/" + strName + "/master/README.rst"}

    @staticmethod
    def setWithToken(bWithToken):
        if bWithToken is not InputOutputAgent.__bWithToken:
            InputOutputAgent.__bWithToken = bWithToken
            InputOutputAgent.__bWithTokenUpdated = True
            # if InputOutputAgent.__gh:
            #     InputOutputAgent.__gh.close()         # there is no .close() method
            InputOutputAgent.__connectToGitHub()

    @staticmethod
    def __connectToGitHub():
        if InputOutputAgent.__gh is None or InputOutputAgent.__bWithTokenUpdated:
            InputOutputAgent.__bWithTokenUpdated = False
            if InputOutputAgent.__bWithToken:
                # the TokenGithubAPI is stored as an environment-variable
                InputOutputAgent.__gh = login(token=str(os.environ['TokenGithubAPI']))
                print('GithubToken is used for connection')
            else:
                InputOutputAgent.__gh = GitHub()
                print('No GithubToken is used for connection')

            # https://github3py.readthedocs.io/en/master/
            # InputOutputAgent.gh.refresh()
            # InputOutputAgent.gh.refresh(True)  # Will send the GET with a header such that if nothing
            # has changed, it will not count against your ratelimit
            # otherwise you'll get the updated user object.

            # get rate limit information
            rates = InputOutputAgent.__gh.rate_limit()
            print('normal ratelimit info: ', rates['resources']['core'])  # => your normal ratelimit info
            print('search ratelimit info: ', rates['resources']['search'])  # => your search ratelimit info

    def loadJSONdata(self, strPathJSON):
        """
        loads the requested json-data either from a file or alternatively from the web
        files are exported in the './json/' directory if they were requested
        """

        jsonAPI = None

        # check if the json file has already been requested and was saved
        if os.path.isfile(strPathJSON):
            # read from it
            with open(strPathJSON) as jsonData:
                if jsonData is None:
                    print("jsonData=None exception: ", strPathJSON)
                jsonAPI = json.load(jsonData)
        else:
            InputOutputAgent.__connectToGitHub()
            repo = InputOutputAgent.__gh.repository(self.strUser, self.strName)
            jsonAPI = repo.as_dict()  # .as_json() returns json.dumps(obj.as_dict())

            # export to json-file
            with open(strPathJSON, 'w') as outfile:
                json.dump(jsonAPI, outfile)
                print('json-data was exported to: ', strPathJSON)

        return jsonAPI, self.strAPIUrl, self.lstReadmePath


    # Get content from readme as string
    def getReadme(self, strPathReadme):

        # Create readme directory
        if not os.path.exists(strPathReadme):
            os.makedirs(strPathReadme)

        strPathReadme += '\\' + self.strUser + '_' + self.strName + '.txt'

        # Check if readme exists already. If so, open it.
        if os.path.isfile(strPathReadme):
            #print("Open readme..." )
            return open(strPathReadme).read()

        else:
            InputOutputAgent.__connectToGitHub()
            #print("Get readme...")

            repo = InputOutputAgent.__gh.repository(self.strUser, self.strName)
            code64readme = repo.readme().content

            # If the content of the received readme is a string and not a NullObject create
            # a new file in directory. Otherwise create an empty file to prevent checking a
            # repo twice.
            if isinstance(code64readme, str):
                strReadme = str(base64.b64decode(code64readme))

            else:
                strReadme = ""

            file = open(strPathReadme, "w")
            file.write(strReadme)
            print('readme was exported to: ', strPathReadme)

            return strReadme



