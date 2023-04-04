# cs4391_project

CS 4391 Spring 2023
Term Project
Due Date: Sunday - May 9th, 2023 11:59 PM

The goal for this project is to combine image processing, feature extraction, clustering and classification methods we have discussed in class to achieve basic scene understanding. 
Specifically, we will examine the task of scene recognition starting with very simple methods – small-scale images and nearest neighbor classification -- and then move on to quantized local features and linear classifiers learned by support vector machines.
For this project, you will be given two sets of images: Train and Test. In each set of images, you can find a total of 3 categories of images (3 different kind of scenes). The size of images is relatively small (all of them are ~250*250 in size) and mostly grayscale images. 
Below are two sample images from the sets: (left) An outdoor scene in the forest; (right) An indoor scene of bedroom.
   

Project Requirement: 
NOTE – if you are working on this project INDIVIDUALLY, you can SKIP the requirements that are colored BLUE. 
1.	For ALL training images, do the following pre-processing: 
a.	Convert to grayscale images, and adjust the brightness if necessary (e.g. if average brightness is less than 0.4, increase brightness; if average brightness is greater than 0.6, reduce brightness)
b.	Resize the image to TWO different sizes: 200*200 and 50*50 and save them. 
2.	Extract SIFT features on ALL training images and save the data.
3.	Extract Histogram features on ALL training images and save the data. 
4.	Perform the following FOUR TRAINING on the data: 
a.	Represent the image directly using the 50*50 (2500) pixel values and use the Nearest Neighbor classifier 
b.	Represent the image using SIFT feature data and use Nearest Neighbor classifier
c.	Represent the image using histogram feature data and use Nearest Neighbor classifier
d.	Represent the image using SIFT feature data and use linear SVM classifier
5.	Test the THREE(/TWO if you worked individually) trained classifiers using ALL test images and report the following results: 
a.	percentage of correctly classified images in the test set
b.	percentage of False Positive (images that are falsely classified)
c.	percentage of False Negative (images that are not classified)
6.	Write UP – please generate a concise report for the project. In the report you will describe how you implemented the project; and report the results in step 5. Also in the report, please briefly analyze and discuss the results you get from step 5 (e.g. comparisons, why one method performs better than the other) and any other findings (e.g. what affected the accuracy). 
7.	(BONUS) Experiment with extracting ANY other feature to represent the image; and then train/test on the dataset. Record and discuss the result in the write up report.

Submission Instructions: 
Please submit your source code and report to eLearning ONLY. 
![image](https://user-images.githubusercontent.com/71054249/229935880-de9b5452-bbd9-41d6-ac7d-d0f8cb9ee98d.png)
