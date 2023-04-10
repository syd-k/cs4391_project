# Huy P Nguyen, Sydney Khamphouseng
# CV Project

import cv2 as cv
import numpy as np
import sys
from os import listdir, mkdir
from os.path import isfile, join

# return a list of gray images from a chosen directory
def convImagesToGray(path):
    # read all images in a folder
    onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
    onlyfiles.sort()
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread( join(path,onlyfiles[n]) )

    grays = []  # a list of gray images
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to gray scale
        grays.append(gray)
    
    return grays

# convert to gray images
grays1 = convImagesToGray('ProjData/Train/bedroom/') 
grays2 = convImagesToGray('ProjData/Train/Coast/') 
grays3 = convImagesToGray('ProjData/Train/Forest/') 
total_grays = grays1 + grays2 + grays3    # a list of ALL training gray images

# create relevant new folders
try: 
    mkdir('resize200') 
    mkdir('resize50')
except OSError as error: pass

try: 
    mkdir('dark_image')
    mkdir('bright_image')
except OSError as error: pass

# adjust brightness
count = 0
grays = []  # a list of adjusted gray images
for gray in total_grays:
    gray_row, gray_col = gray.shape

    sum = np.sum(gray)
    mean = sum / (gray_row*gray_col)
    avg_brightness = mean / 255

    if avg_brightness < 0.4:
        # make image brighter
        diff = 0.4 - avg_brightness
        beta = diff * 256
        brighter_image = cv.convertScaleAbs(gray, alpha=1, beta=beta)
        cv.imwrite('dark_image/' + str(count) + '_before.jpg', gray)
        cv.imwrite('dark_image/' + str(count) + '.jpg', brighter_image)
        grays.append(brighter_image)


    elif avg_brightness > 0.6: 
        # make image darker
        diff = avg_brightness - 0.6
        beta = -diff * 256
        darker_image = cv.convertScaleAbs(gray, alpha=1, beta=beta)
        cv.imwrite('bright_image/' + str(count) + '_before.jpg', gray)
        cv.imwrite('bright_image/' + str(count) + '.jpg', darker_image)
        grays.append(darker_image)
    
    else: grays.append(gray)
    count += 1

# resize the gray images
grays200 = []   # a list of gray images size 200x200
grays50 = []    # a list of gray images size 50x50

count = 0
for gray in grays:
    g200 = cv.resize(gray, (200, 200))
    g50 =  cv.resize(gray, (50, 50))
    grays200.append(g200)
    grays50.append(g50)

    cv.imwrite('resize200/' + str(count) + '.jpg', g200)
    cv.imwrite('resize50/' + str(count) + '.jpg', g50)
    count += 1

