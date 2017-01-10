''' This file contains tuples of all relevant features which are used to distinguish two repositories '''
from collections import namedtuple
# namedtuple are used as replacement for struct-like objects
# but the attributes are unmutable

# ~~~~~ define the features which are used for training and classification ~~~~~~
IntFeatures = namedtuple('IntFeatures',
                         # 'iNumContributors' -> the url for the contributors doesn't has the actual size
                         'iSubscriberCount '  # addition
                         'dRepoActivity ' # old name 'dCodeFrequency '
                         'dCommitIntervals '
                         'iWatchersCount '  #'iNumBranches '
                         'iOpenIssues '
                         'iDevTime '
                         'iSize'  # addition
                         )

StringFeatures = namedtuple('StringFeatures',
                            'strTitle'
                            'strDescription'
                            'strReadmeContent'
                            'strLanguages'
                            'strFolderNames')
