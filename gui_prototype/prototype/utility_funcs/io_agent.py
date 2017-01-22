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
    __bRedownload = False

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
    def setRedownload(bRedownload):
        """
        sets up if the readme and json-file shall get redownload

        :param bRedownload: true, false
        :return:
        """
        InputOutputAgent.__bRedownload = bRedownload


    @staticmethod
    def setWithToken(bWithToken):
        """
        sets up if the github token shall be used for connection to github

        :param bWithToken: true, false
        :return:
        """
        if bWithToken is not InputOutputAgent.__bWithToken:
            # if InputOutputAgent.__gh:
            #     InputOutputAgent.__gh.close()         # there is no .close() method
            try:
                InputOutputAgent.__connectToGitHub(bWithToken)
                InputOutputAgent.__bWithToken = bWithToken
                InputOutputAgent.__bWithTokenUpdated = True
            except Exception as e:
                raise e

    @staticmethod
    def __connectToGitHub(bWithToken):
        """
        private method to establish a connection to github

        :param bWithToken: true, false
        :return:
        """
        if InputOutputAgent.__gh is None or InputOutputAgent.__bWithTokenUpdated:
            if bWithToken:
                # the TokenGithubAPI is stored as an environment-variable
                try:
                    InputOutputAgent.__gh = login(token=str(os.environ['TokenGithubAPI']))
                    InputOutputAgent.__bWithTokenUpdated = False
                    print('GithubToken is used for connection')

                except Exception as ex:
                    raise ConnectionError('no connection to GitHub could be established')
            else:
                try:
                    InputOutputAgent.__gh = GitHub()
                    InputOutputAgent.__bWithTokenUpdated = False
                    print('No GithubToken is used for connection')
                except Exception as ex:
                    raise ConnectionError('no connection to GitHub could be established')

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
        if os.path.isfile(strPathJSON) and InputOutputAgent.__bRedownload is False:
            # read from it
            print("[INFO] Using locally cached version of repository")
            with open(strPathJSON) as jsonData:
                try:
                    if jsonData is None:
                        print("jsonData=None exception: ", strPathJSON)
                    jsonAPI = json.load(jsonData)
                except Exception as ex:
                    raise ImportError('the json-data couldn\'t be loaded from the file: ' + strPathJSON)
                    raise ex
        else:
            InputOutputAgent.__connectToGitHub(InputOutputAgent.__bWithToken)
            repo = InputOutputAgent.__gh.repository(self.strUser, self.strName)

            if repo:
                jsonAPI = repo.as_dict()  # .as_json() returns json.dumps(obj.as_dict())

                # export to json-file
                with open(strPathJSON, 'w') as outfile:
                    json.dump(jsonAPI, outfile)
                    print('json-data was exported to: ', strPathJSON)
            else:
                raise ConnectionError('the given repository is not accessible')

        return jsonAPI, self.strAPIUrl, self.lstReadmePath


    def getReadme(self, strPathReadme):
        """
        Gets the content from the Redme as a string.
        The Readme is either loaded from file or web.

        :param strPathReadme: path were the readme is loaded and exported to
        :return:
        """

        # Create readme directory
        if not os.path.exists(strPathReadme):
            os.makedirs(strPathReadme)

        strPathReadme += '\\' + self.strUser + '_' + self.strName + '.txt'

        # Check if readme exists already. If so, open it.
        if os.path.isfile(strPathReadme) and InputOutputAgent.__bRedownload is False:
            return open(strPathReadme).read()

        else:
            InputOutputAgent.__connectToGitHub(InputOutputAgent.__bWithToken)

            repo = InputOutputAgent.__gh.repository(self.strUser, self.strName)
            code64readme = repo.readme().content

            # If the content of the received readme is a string and not a NullObject create
            # a new file in directory. Otherwise create an empty file to prevent checking a
            # repo twice.
            if repo:
                if isinstance(code64readme, str):
                    strReadme = str(base64.b64decode(code64readme))

                else:
                    strReadme = ""

                file = open(strPathReadme, "w")
                file.write(strReadme)
                print('readme was exported to: ', strPathReadme)
            else:
                raise ConnectionError('the given repository is not accessible')

            return strReadme
