import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
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
grays1 = convImagesToGray('/ProjData/Train/bedroom') 
grays2 = convImagesToGray('/ProjData/Train/Coast') 
grays3 = convImagesToGray('/ProjData/Train/Forest') 
total_grays = grays1 + grays2 + grays3    # a list of ALL training gray images

def histogram(total_grays):
    try: 
        mkdir('histograms')
    except OSError as error: 
        pass

    for i in range(len(total_grays)):
        plt.hist(total_grays[i].ravel(), 256, (0, 256))
        plt.savefig(f'histograms/{i}.jpg')
        plt.clf()

histogram(total_grays)
    






