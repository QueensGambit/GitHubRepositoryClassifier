from bs4 import BeautifulSoup
import re
import nltk
import mimetypes
from nltk.corpus import stopwords   # Import the stop word list
from nltk.stem.porter import PorterStemmer

# refine the input string
def prepare_words(raw_text, bApplyStemmer=True, bCheckStopWords=False):
    """
    prepares the word for the comparision with the vocab list

    :param raw_text: text with control characters, number,
    :param bApplyStemmer: true if is stemming shall be applied
    :param bCheckStopWords: true if stopwords shall be removed
    :return: normed word list
    """

    raw_text = re.sub(r'^http?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)                     # remove web-adresses
    raw_text = re.sub(r'\\.', ' ', raw_text)                              # remove all control-characters: \n, \t ...
    # http://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python

    raw_text = re.sub(r'\([^()]*\)', ' ', raw_text)

    letters = re.sub("[^a-zA-Z]", " ", raw_text)                        # remove everything that isn't a letter

    words = letters.lower().split()                                     # write words into array

    if bCheckStopWords:
        words = [w for w in words if w not in stopwords.words("english") and w not in stopwords.words("german")]   # remove "filler" words

    if bApplyStemmer:
        # see: http://www.nltk.org/howto/stem.html for more details
        stemmer = PorterStemmer()
        singles = [stemmer.stem(word) for word in words]   # only allow words with a length higher than 2  if len(word) > 2
        singles = [single for single in singles if len(single) > 2]
        words = " ".join(singles)

    return words                                             # return the words as a string, separator: space


def validate_url(url_in):
    """
    Performs some simple string checks to validate the URL for further processing

    :param url_in: The URL to perform the checks on
    :return: error: errorcode
    """
    if url_in == "":
        error = "[ERROR] Input is empty"
        return False
    elif not url_in.startswith("https://"):
        error = "[ERROR] Input doesn't start with https://"
        return False
    elif not url_in.startswith("https://github.com/"):
        error = "[ERROR] Input is not a GitHub URL"
        return False
    else:
        error = "[INFO] Input is a valid URL"
        return True

def validate_txtfile(path):
    """
    Checks file type whether its txt or not
    :param path: path to file
    :return:
    """
    bFile = True if mimetypes.guess_type(path)[0] == 'text/plain' else False
    return bFile
