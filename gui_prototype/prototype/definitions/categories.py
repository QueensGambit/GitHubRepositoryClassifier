from enum import IntEnum

class Category(IntEnum):
    DEV = 0,
    HW = 1,
    EDU = 2,
    DOCS = 3,
    WEB = 4,
    DATA = 5,
    OTHER = 6

class CategoryStr():
    lstStrCategories = ['DEV', 'HW', 'EDU', 'DOCS', 'WEB', 'DATA', 'OTHER']
    lstStrIcons = [None] * len(lstStrCategories)
    # set the icons for th each category -> this can be adjusted
    lstStrIcons[Category.DEV] = "code_v2_-8x.png"  # "bug-8x.png" #""code-8x.png"
    lstStrIcons[Category.HW] = "home-8x.png"
    lstStrIcons[Category.EDU] = "bullhorn-8x.png"
    lstStrIcons[Category.DOCS] = "document-8x.png"  # "clipboard-8x.png" #"book-8x.png"
    lstStrIcons[Category.WEB] = "globe-8x.png"
    lstStrIcons[Category.DATA] = "cloud-download-8x.png"
    lstStrIcons[Category.OTHER] = "beaker-8x.png"

    lstStrColors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'gray', 'lightblue', 'tomato']




