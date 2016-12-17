from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords   # Import the stop word list


# refine the input string
def prepare_words(raw_text):

    raw_text = raw_text[1:]
    raw_text = raw_text.rstrip()                                        # remove all control-characters: \n, \t ...
    beautiful = BeautifulSoup(raw_text, "lxml")                         # remove all html tags
    letters = re.sub("[^a-zA-Z]", " ", beautiful.get_text())            # remove everything that isn't a letter
    words = letters.lower().split()                                     # write words into array
    words = [w for w in words if not w in stopwords.words("english")]   # remove "filler" words

    return " ".join(words)                                              # return the words as a string, separator: space
