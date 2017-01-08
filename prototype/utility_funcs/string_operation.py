from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords   # Import the stop word list
from nltk.stem.porter import PorterStemmer

# refine the input string
def prepare_words(raw_text):

    # raw_text = raw_text[1:]                                             # remove first letter
    raw_text = raw_text.rstrip()                                        # remove all control-characters: \n, \t ...
    beautiful = BeautifulSoup(raw_text, "lxml")                         # remove all html tags
    letters = re.sub("[^a-zA-Z]", " ", beautiful.get_text())            # remove everything that isn't a letter
    words = letters.lower().split()                                     # write words into array
    words = [w for w in words if not w in stopwords.words("english")]   # remove "filler" words

    # see: http://www.nltk.org/howto/stem.html for more details
    stemmer = PorterStemmer()
    singles = [stemmer.stem(word) for word in words]

    # return words
    return " ".join(singles)                                              # return the words as a string, separator: space
