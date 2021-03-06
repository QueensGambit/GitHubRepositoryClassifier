''' This script is a combination of the functionality of github_api_via_json.py  and wordcloud/masked.py
in order to visualize the readme-files of the example repositories'''

from os import path

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from PIL import ImageOps
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread

from definitions.categories import Category

# define the path where the images are stored
d = path.dirname(__file__)
d += '/icons/'


iNumCategories = 7
lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHERS']

# initialize a list with None-values
lstReadmeURL = [None]*iNumCategories


# Set a readme-file for each category
# Dev example
lstReadmeURL[Category.DEV] = 'https://raw.githubusercontent.com/scipy/scipy/master/README.rst'
# Homework example
lstReadmeURL[Category.HW] = 'https://raw.githubusercontent.com/uwhpsc-2016/example-python-homework/master/README.md'
# Edu example
lstReadmeURL[Category.EDU] = 'https://raw.githubusercontent.com/AllThingsSmitty/jquery-tips-everyone-should-know/master/README.md'
# Docs example
lstReadmeURL[Category.DOCS] = 'https://raw.githubusercontent.com/raspberrypi/documentation/master/README.md'
# Web example
lstReadmeURL[Category.WEB] = 'https://raw.githubusercontent.com/JaceRobinson8/jacerobinson8.github.io/master/README.md'
# Data example
lstReadmeURL[Category.DATA] = 'https://raw.githubusercontent.com/OpenExoplanetCatalogue/open_exoplanet_catalogue/master/README.md'
# Other example
lstReadmeURL[Category.OTHER] = 'https://raw.githubusercontent.com/restic/others/master/README.md'


# here the readme-contents will be stored
lstReadmeText = []

for strURL in lstReadmeURL:
    r = requests.get(strURL)
    lstReadmeText.append(r.text)


# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
#img_mask = np.array(Image.open(path.join(d, "cloud_icon.png")))

# define the image height and width for the word cloud image exports
iWidth = 512
iHeight = 512

# DEV, HW, EDU, DOCS, WEB, DATA
lstStrIcons = [None]*iNumCategories


# set the icons for th each category -> this can be adjusted
lstStrIcons[Category.DEV] = "code_v2_-8x.png" #"bug-8x.png" #""code-8x.png"
lstStrIcons[Category.HW] = "home-8x.png"
lstStrIcons[Category.EDU] = "bullhorn-8x.png"
lstStrIcons[Category.DOCS] = "document-8x.png" #"clipboard-8x.png" #"book-8x.png"
lstStrIcons[Category.WEB] = "globe-8x.png"
lstStrIcons[Category.DATA] = "cloud-download-8x.png"
lstStrIcons[Category.OTHER] = "beaker-8x.png"


lstImgMask = []
for strIconPath in lstStrIcons:

    # you can split an image into different channels
    # needed is the alpha channel only
    # red, green, blue, alpha = img.split()
    # alpha = img.split()[-1]
    img = (Image.open(path.join(d, strIconPath))).split()[-1]

    # the mask is inverted, so invert it again
    img = ImageOps.invert(img)
    img = img.resize((iWidth, iHeight), Image.NONE)
    lstImgMask.append(np.array(img))


stopwords = set(STOPWORDS)
stopwords.add("said")

lstWC = []

strPath = path.dirname(__file__)

image_colors = None

# test about coloring the image
"""
# imgGrayVariance = np.array(Image.open(strPath + "/icons/" + "gray_variance.png"))  # imread(strPath + "/icons/" + "color_variance.png")
imgGrayVariance = Image.open(strPath + "/icons/" + "gray_variance.png")

# imgGrayVariance += (u_int8)(40, 23, 159) #Color(CategoryStr.lstStrColors[iLabel]).to_rgb()
# http://stackoverflow.com/questions/6483489/change-the-color-of-all-pixels-with-another-color
r2, g2, b2 = 40, 23, 159
color_mod = 40, 23, 159

width, height = imgGrayVariance.size

pix = imgGrayVariance.load()

for x in range(0, width):
    for y in range(0, height):
        # new_color = pix[x, y] + color_mod
        old_color = pix[x, y]
        new_color = tuple([sum(x) for x in zip(old_color,color_mod)]) #old_color + color_mod
        # im.putpixel((x,y), )
        # if pix[x, y] == old_color:
        imgGrayVariance.putpixel((x, y), new_color)

# r, g, b = imgGrayVariance[:,:,0], imgGrayVariance[:,:,1], imgGrayVariance[:,:,2]
# imgGrayVariance[:,:,:3][:] += int([r2, g2, b2])
fig = plt.figure()
plt.imshow(imgGrayVariance)

imgGrayVariance = np.array(imgGrayVariance)
# create coloring from image
image_colors = ImageColorGenerator(imgGrayVariance)
"""

for i in range(0, iNumCategories):

    wc = WordCloud(background_color="white", max_words=2000, mask=lstImgMask[i], #imgGrayVariance,#
    stopwords=stopwords)
    lstWC.append(wc) #, width=512, height=512))

i = 0

# define the path where the wordclouds images will be exported
strPathExport = path.dirname(__file__)
strPathExport += '/export/'
for wc in lstWC:
    # generate word cloud
    wc.generate(lstReadmeText[i])
    # wc.recolor(color_func=image_colors)

    strFileName = "WordCloud_" + lstStrCategories[i] + ".png"
    # store to file
    wc.to_file(path.join(strPathExport, strFileName))

    # show
    fig = plt.figure()
    fig.canvas.set_window_title(strFileName)

    # setting the title is optional
    #plt.title(lstStrCategories[i])
    plt.imshow(wc)
    plt.axis("off")

    # show the mask is optional
    #plt.figure()
    #plt.imshow(lstImgMask[0], cmap=plt.cm.gray)
    #plt.axis("off")

    i += 1

plt.show()
