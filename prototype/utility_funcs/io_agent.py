'''
@file: io_funcs.py
Created on 11.12.2016 19:59
@project: GitHubRepositoryClassifier

@author: QueensGambit
SEE LICENSE.TXT
'''
from clyent import json_help

import json
import requests
import base64

# for installtion use:
# pip install github3.py
from github3 import GitHub
from github3 import login
import os


class InputOutputAgent:

    def __init__(self, strUser, strName, bWithToken=False):
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
        self.bWithToken = bWithToken

        if self.bWithToken:
            # the TokenGithubAPI is stored as an environment-variable
            self.gh = login(token=str(os.environ['TokenGithubAPI']))


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
                jsonAPI = json.load(jsonData)
        else:

            if self.bWithToken:
                repo = self.gh.repository(self.strUser, self.strName)

                jsonAPI = repo.as_dict()  # .as_json() returns json.dumps(obj.as_dict())

            else:
                jsonAPI = (requests.get(self.strAPIUrl)).json()

            # export to json-file
            with open(strPathJSON, 'w') as outfile:
                json.dump(jsonAPI, outfile)
                print('json-data was exported to: ', strPathJSON)

        return jsonAPI, self.strAPIUrl, self.lstReadmePath


    # get String from readme file
    def getREADME(self, strPathReadme):

        filename = '\\' + self.strUser + self.strName + '.txt'
        strPathReadme += filename

        if os.path.isfile(strPathReadme):
            print("Open " + strPathReadme)
            return open(strPathReadme).read()

        else:
            print("Try readme...")

            repo = self.gh.repository(self.strUser, self.strName)
            code64readme = repo.readme().content

            if isinstance(code64readme, str):
                strReadme = str(base64.b64decode(code64readme))
                f = open(strPathReadme, "w")
                f.write(strReadme)
                return strReadme

            else:
                return "No readme in repo"





