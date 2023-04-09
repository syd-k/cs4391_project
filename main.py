# Huy P Nguyen, Sydney Khamphouseng
# CV Project

import cv2 as cv
import numpy as np
import sys
from os import listdir, mkdir
from os.path import isfile, join

# return a list of gray images from a chosen directory
def convImagesToGray(pathToDirectory):
    # read all images in a folder
    mypath='ProjData/Train/bedroom/'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread( join(mypath,onlyfiles[n]) )

    grays = []  # a list of gray images
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to gray scale
        grays.append(gray)
    
    return grays

grays1 = convImagesToGray('ProjData/Train/bedroom/') 
grays2 = convImagesToGray('ProjData/Train/Coast/') 
grays3 = convImagesToGray('ProjData/Train/Forest/') 
grays = grays1 + grays2 + grays3    # a list of ALL training gray images

# resize
grays200 = []   # a list of gray images size 200x200
grays50 = []    # a list of gray images size 50x50

# create new folders for resize images
try: 
    mkdir('resize200') 
    mkdir('resize50')
except OSError as error: pass

count = 0
for gray in grays:
    g200 = cv.resize(gray, (200, 200))
    g50 =  cv.resize(gray, (50, 50))
    grays200.append(g200)
    grays50.append(g50)

    cv.imwrite('resize200/' + str(count) + '.jpg', g200)
    cv.imwrite('resize50/' + str(count) + '.jpg', g50)
    count += 1


list_brightness = []    # a list of avg brightness in grays200
for gray in grays200:
    gray_row, gray_col = gray.shape

    sum = np.sum(gray)
    mean = sum / (gray_row*gray_col)
    avg_brightness = mean / 255
    list_brightness.append(avg_brightness)

print(len(list_brightness))
print(f"min brightness = {min(list_brightness)}")
print(f"max brightness = {max(list_brightness)}")


