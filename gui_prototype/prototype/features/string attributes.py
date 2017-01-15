"""
@file: string attributes.py
Created on 16.12.2016 11:03
@project: GitHubRepositoryClassifier

@author: Plagiatus

Handles the initial processing as well as the individual processing of the string attributes of a repository,
including README, title, description
"""
from os import path

class StringAttributesAgent:

    def __init__(self):
        pass
        # nothing here

    def preProcessTitles(self):             # TODO: read all titles in the saved JSONs into a SparesMatrix
        strPath = path.dirname(__file__)


    def processTitle(self, strPathJSON):    # TODO: return the title in the base of the overall SparesMatrix
        pass



    def preProcessDescriptions(self):       # TODO: read all descriptions in the saved JSONs into a SparesMatrix
        pass



    def processDescription(self, strPathJSON): # TODO: return the description in the base of the overall SparesMatrix
        pass



    def preProcessReadmes(self):            # TODO: read all saved Readmes into a SparesMatrix
        pass



    def processReadme(self, strPathReadme): # TODO return the readme in the base of the overall SparesMatrix
        pass