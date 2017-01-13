from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords   # Import the stop word list
from nltk.stem.porter import PorterStemmer

# refine the input string
def prepare_words(raw_text, bApplyStemmer=True):

    raw_text = re.sub(r'^http?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)                     # remove web-adresses
    # raw_text = raw_text[1:]                                             # remove first letter
    raw_text = re.sub(r'\\.', ' ', raw_text)                              # remove all control-characters: \n, \t ...
    # http://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    # mpa = dict.fromkeys(range(32))
    # >> > 'abc\02de'.translate(mpa)
    # raw_text = raw_text.rstrip()                                        # remove all control-characters: \n, \t ...
    # raw_text = raw_text.translate(mpa)                                    # remove all control-characters: \n, \t ...

    # raw_text = re.sub(r"[\{\(\[\<].*?[\)\]\>\}]", "", raw_text)         # remove everything in a bracket
    raw_text = re.sub(r'\([^()]*\)', ' ', raw_text)
    # http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    raw_text = re.sub(r'\[[^()]*\]', ' ', raw_text)
    # raw_text = re.sub(r'\<[^()]*\>', ' ', raw_text)

    # beautiful = BeautifulSoup(raw_text, "lxml")                         # remove all html tags
    # letters = re.sub("[^a-zA-Z]", " ", beautiful.get_text())            # remove everything that isn't a letter

    letters = re.sub("[^a-zA-Z]", " ", raw_text)                        # remove everything that isn't a letter

    words = letters.lower().split()                                     # write words into array
    words = [w for w in words if not w in stopwords.words("english")]   # remove "filler" words

    if bApplyStemmer:
        # see: http://www.nltk.org/howto/stem.html for more details
        stemmer = PorterStemmer()
        singles = [stemmer.stem(word) for word in words]   # only allow words with a length higher than 2  if len(word) > 2
        singles = [single for single in singles if len(single) > 2]
        words = " ".join(singles)

    # return words
    return words                                             # return the words as a string, separator: space
