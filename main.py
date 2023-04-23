# Huy P Nguyen, Sydney Khamphouseng
# CV Project

import cv2 as cv
import numpy as np
import sys
from os import listdir, mkdir
from os.path import isfile, join
import matplotlib.pyplot as plt

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

#-----------SECTION: Part 1 PREPROCESS TRAINING DATA------------------------------
# convert to gray images
grays1 = convImagesToGray('ProjData/Train/bedroom/') 
grays2 = convImagesToGray('ProjData/Train/Coast/') 
grays3 = convImagesToGray('ProjData/Train/Forest/') 
total_grays = grays1 + grays2 + grays3    # a list of ALL training gray images

# labels for the training images. 0: bedroom, 1: Coast, 2: Forest
train_labels = np.zeros(300)
train_labels[100:200] = 1
train_labels[200:300] = 2
train_labels = np.array(train_labels).astype(np.float32)
train_labels = train_labels.reshape(300,1)

# create relevant new folders
try: 
    mkdir('resize200') 
    mkdir('resize50')
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
        grays.append(brighter_image)


    elif avg_brightness > 0.6: 
        # make image darker
        diff = avg_brightness - 0.6
        beta = -diff * 256
        darker_image = cv.convertScaleAbs(gray, alpha=1, beta=beta)
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


#-----------PREPROCESS TEST DATA-------------------------------------------
# convert to gray images
test_grays1 = convImagesToGray('ProjData/Test/bedroom/') 
test_grays2 = convImagesToGray('ProjData/Test/Coast/') 
test_grays3 = convImagesToGray('ProjData/Test/Forest/') 
test_total_grays = test_grays1 + test_grays2 + test_grays3    # a list of ALL training gray images

# labels for the test images. 0: bedroom, 1: Coast, 2: Forest
test_labels = np.zeros(604)
test_labels[116:376] = 1
test_labels[376:604] = 2
test_labels = np.array(test_labels).astype(np.float32)
test_labels = test_labels.reshape(604,1)

# adjust brightness on test images
test_grays = []  # a list of adjusted gray images
for gray in test_total_grays:
    gray_row, gray_col = gray.shape

    sum = np.sum(gray)
    mean = sum / (gray_row*gray_col)
    avg_brightness = mean / 255

    if avg_brightness < 0.4:
        # make image brighter
        diff = 0.4 - avg_brightness
        beta = diff * 256
        brighter_image = cv.convertScaleAbs(gray, alpha=1, beta=beta)
        test_grays.append(brighter_image)


    elif avg_brightness > 0.6: 
        # make image darker
        diff = avg_brightness - 0.6
        beta = -diff * 256
        darker_image = cv.convertScaleAbs(gray, alpha=1, beta=beta)
        test_grays.append(darker_image)
    
    else: test_grays.append(gray)

# resize the test gray images
test_grays200 = []   # a list of gray images size 200x200
test_grays50 = []    # a list of gray images size 50x50

for gray in test_grays:
    g200 = cv.resize(gray, (200, 200))
    g50 =  cv.resize(gray, (50, 50))
    test_grays200.append(g200)
    test_grays50.append(g50)


# -----SECTION: Part 2 SIFT features-------------------------------------------

sift = cv.SIFT_create()
NOkp = []                                       # total number of keypoints for ALL training data
for i in range(len(grays200)):
    kp, des = sift.detectAndCompute(grays200[i],None)
    NOkp.append(len(kp))

    if i == 0:
        allDescriptors = np.array(des).astype(np.float32)
    else: allDescriptors = np.concatenate((allDescriptors, des), axis=0)

# define criteria, number of clusters(K)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 50

# apply kmeans()
compactness,label,region=cv.kmeans(allDescriptors,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
label = np.array(label).astype(np.float32)
region = np.array(region).astype(np.float32)
region_labels = np.arange(K).astype(np.float32)

train = []                                      # 300 samples x 50 sift features
kp_count = 0
for i in range(len(NOkp)):
    # img 0
    siftFeat = np.zeros(K)
    for j in range(0, NOkp[i]):
        siftFeat[int (label[kp_count + j, 0]) ] += 1

    siftFeat2 = np.copy(siftFeat)
    train.append(siftFeat2)

    kp_count += NOkp[i]

train_sift = np.array(train).astype(np.float32)
try: 
    mkdir('SIFT')
except OSError as error: pass

for i in range(len(train_sift)):
    np.savetxt('SIFT/siftData' + str(i) +'.csv', train_sift[i], delimiter=',')


# -------SECTION: Part 3 Histogram Features---------------------------------------

def histogram(grays):
    hist_feat = []
    for i in range(len(grays)):
        hist = cv.calcHist([grays[i]],[0],None,[256],[0,256])
        hist = np.array(hist).astype(np.float32)
        hist = np.transpose(hist)
        hist_feat.append(hist[0])

    return hist_feat

hist_feat = histogram(grays200)
hist_feat = np.array(hist_feat).astype(np.float32)

try: 
    mkdir('histograms')
except OSError as error: pass

for i in range(len(hist_feat)):
    np.savetxt('histograms/histData' + str(i) +'.csv', hist_feat[i], delimiter=',')


# -------SECTION: Part 4a Classifier------------------------------------------------

# part 1: 50x50 images flatten into 1x2500 matrix 
train = []
for gray in grays50:
    train.append(gray.flatten())

train = np.array(train).astype(np.float32)

test = []
for gray in test_grays50:
    test.append(gray.flatten())

test = np.array(test).astype(np.float32)

# Initiate kNN, train it on the training data, then test it with the test data with k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=1)

# Now we check the accuracy of classification
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size

print("Part 4a: Using nearest neighbor classifier with 50x50 pixel values")
print(f"accuracy = {accuracy:.2f}%\n")

res = np.transpose(result)[0]

def stats(res):
    #-----False Positive------------------------------------------------------
    # label 0: bedroom
    NOfp_label0 = 0                    # number of false positive for label 0
    for i in range(116, len(res)):
        if res[i] == 0: NOfp_label0 += 1

    # label 1: Coast
    NOfp_label1 = 0                    # number of false positive for label 1
    for i in range(0, 116):
        if res[i] == 1: NOfp_label1 += 1

    for i in range(376, len(res)):
        if res[i] == 1: NOfp_label1 += 1

    # label 2: Forest
    NOfp_label2 = 0                    # number of false positive for label 2
    for i in range(376):
        if res[i] == 2: NOfp_label2 += 1

    #-----False Negative------------------------------------------------------
    # bedroom
    NOfn_label0 = 0                     # number of false negative for label 0
    for i in range(116):
        if not res[i] == 0: NOfn_label0 += 1

    # label 1: Coast
    NOfn_label1 = 0                     # number of false negative for label 1
    for i in range(116, 376):
        if not res[i] == 1: NOfn_label1 += 1

    # label 2: Forest
    NOfn_label2 = 0                     # number of false negative for label 2
    for i in range(376, len(res)):
        if not res[i] == 2: NOfn_label2 += 1

    print("Bedroom:")
    print(f"False Positive = {100*NOfp_label0/len(res):.2f}%")
    print(f"False Negative = {100*NOfn_label0/len(res):.2f}%")
    print()
    print("Coast:")
    print(f"False Positive = {100*NOfp_label1/len(res):.2f}%")
    print(f"False Negative = {100*NOfn_label1/len(res):.2f}%")
    print()
    print("Forest:")
    print(f"False Positive = {100*NOfp_label2/len(res):.2f}%")
    print(f"False Negative = {100*NOfn_label2/len(res):.2f}%")
    print("---------------------------------------------------------------------\n")

stats(res)


# -------SECTION: Part 4b classifier------------------------------------------------

test = []
for i in range(len(test_grays200)):
    siftFeat = np.zeros(K)
    kp, des = sift.detectAndCompute(test_grays200[i],None)
    des = np.array(des).astype(np.float32)
    # Initiate kNN, train it on the training data, then test it with the test data with k=1
    knn = cv.ml.KNearest_create()
    knn.train(region, cv.ml.ROW_SAMPLE, region_labels)
    ret,result,neighbours,dist = knn.findNearest(des,k=1)

    for j in range(len(kp)):
        siftFeat[int (result[j, 0]) ] += 1

    siftFeat2 = np.copy(siftFeat)
    test.append(siftFeat2)

test = np.array(test).astype(np.float32)

# Initiate kNN, train it on the training data, then test it with the test data with k=1
knn = cv.ml.KNearest_create()
knn.train(train_sift, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=1)

# Now we check the accuracy of classification
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print("Part 4b: Using nearest neighbor classifier with SIFT feature")
print(f"accuracy = {accuracy:.2f}%\n")

res = np.transpose(result)[0]
stats(res)


# -------SECTION: Part 4c classifier------------------------------------------------

test_hist_feat = histogram(test_grays200)
test_hist_feat = np.array(test_hist_feat).astype(np.float32)

# Initiate kNN, train it on the training data, then test it with the test data with k=1
knn = cv.ml.KNearest_create()
knn.train(hist_feat, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test_hist_feat,k=1)

# Now we check the accuracy of classification
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print("Part 4c: Using nearest neighbor classifier with histogram feature")
print(f"accuracy = {accuracy:.2f}%\n")

res = np.transpose(result)[0]
stats(res)


# -------SECTION: Part 4d classifier------------------------------------------------

# bedroom img is positive label and everything else is negative label
train_labels0 = np.copy(train_labels).astype(np.int32)
train_labels0[:100] = 1
train_labels0[100:] = -1

svm0 = cv.ml.SVM_create()
svm0.setType(cv.ml.SVM_C_SVC)
svm0.setKernel(cv.ml.SVM_LINEAR)
svm0.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm0.train(train_sift, cv.ml.ROW_SAMPLE, train_labels0)

response0 = svm0.predict(train_sift[0:10])[1]

# Coast img is positive label and everything else is negative label
train_labels1 = np.copy(train_labels).astype(np.int32)
train_labels1[:100] = -1
train_labels1[100:200] = 1
train_labels1[200:] = -1

svm1 = cv.ml.SVM_create()
svm1.setType(cv.ml.SVM_C_SVC)
svm1.setKernel(cv.ml.SVM_LINEAR)
svm1.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm1.train(train_sift, cv.ml.ROW_SAMPLE, train_labels1)

response1 = svm1.predict(train_sift[0:10])[1]

# Forest img is positive label and everything else is negative label
train_labels2 = np.copy(train_labels).astype(np.int32)
train_labels2[:200] = -1
train_labels2[200:] = 1

svm2 = cv.ml.SVM_create()
svm2.setType(cv.ml.SVM_C_SVC)
svm2.setKernel(cv.ml.SVM_LINEAR)
svm2.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm2.train(train_sift, cv.ml.ROW_SAMPLE, train_labels2)

response2 = svm2.predict(train_sift[0:10])[1]

response0 = np.transpose(response0)[0]
response1 = np.transpose(response1)[0]
response2 = np.transpose(response2)[0]

pred = []
for i in range(len(response0)):
    if response0[i] == 1:
        pred.append(0)
    elif response1[i] == 1:
        pred.append(1)
    else: pred.append(2)

print("Part 4d: Using SVM with SIFT feature")
print("Prediction for the first 10 training data \n")
print(f"Predicted: {pred}")
print("Actual: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")

