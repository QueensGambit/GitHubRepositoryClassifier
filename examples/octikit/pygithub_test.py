from github import Github

# this is needed for anonymous usage -> alternative login with user account and password
git = Github()
usrQG = git.get_user('QueensGambit')
print(usrQG)
repo = git.get_repo('QueensGambit/Barcode-App')
file_contents = repo.get_file_contents('src/OpenCV_Main.cpp')
print(file_contents)
