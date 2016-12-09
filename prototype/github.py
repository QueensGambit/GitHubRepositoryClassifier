import requests

class GithubRepo:

    def __init__(self, user, title):
        self.user = user
        self.name = title
        self.apiUrl = "https://api.github.com/" + user + "/" + title
        self.lstReadmePath = {"https://raw.githubusercontent.com/" + user + "/" + title + "/master/README.md", "https://raw.githubusercontent.com/" + user + "/" + title + "/master/README.rst"}
        self.apiJSON = None

    def readAttributes(self):
        self.apiJSON = requests.get(self.apiUrl)
