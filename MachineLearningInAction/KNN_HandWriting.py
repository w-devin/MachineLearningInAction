#encoding utf-8
from numpy import *
from PIL import Image
import operator
import os
import sys
import KNN
import cv2

#
# 32 x 32 pictures of handwriting
# Identify the numbers on these pictures
# But the error rate is too high
#

width = 32
height = 32
pixs = width * height

#cast test 330 x 330 image file to text file with opencv 
def image2vector(filename):
	img = cv2.imread(filename)
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	retval, dst = cv2.threshold(grayImage, 125, 255, cv2.THRESH_BINARY)

	#erode
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
	grayImage = cv2.erode(grayImage, kernel)

	[rows, cols] = grayImage.shape
	retVector = zeros((1, pixs))

	for i in range(5,rows - 5, 10):
		for r in range(5,cols - 5, 10):
			count = 0
			for j in range(i - 1, i + 1):
				for k in range(r - 1, r + 1):
					if (dst[j, k] == 0): count += 1

			if(count >= 3):
				retVector[0, (i - 5) / 10 * 32 + (r - 5) / 10] = 1
			else: 
				retVector[0, (i - 5) / 10 * 32 + (r - 5) / 10] = 0

	return retVector


#cast test text file to vector
def img2vector(filename):
	returnVect = zeros((1, pixs))
	fr = open(filename)

	for i in range(height):
		lineStr = fr.readline()
		for j in range(width):
			returnVect[0, width * i + j] = int(lineStr[j])

	return returnVect

def handwritingClassTest():
	trainingPath = r'Datas\digits\trainingDigits'
	testPath = r'Datas\digits\testDigits'

	hwLabels = []
	trainingFileList = os.listdir(trainingPath)
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]

		classNumStr = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector(trainingPath + r'\%s' % fileNameStr)

	testFileList = os.listdir(testPath)
	errorCount = 0.0
	mTest = len(testFileList)

	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]

		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector(testPath + r'\%s' % fileNameStr)
		classifierResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)

		print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr)

		if(classifierResult != classNumStr):
				errorCount += 1.0

	print 'the total number of errors is: %d' % errorCount
	print 'the total error rate is: %f' % (errorCount / float(mTest))

#identify the number in image file
#A bad method to identify the numbers on image
#error rate up to 90% (T.T)
def identify(fileName):
	#training
	trainingPath = r'Datas\digits\trainingDigits'
	hwLabels = []
	trainingFileList = os.listdir(trainingPath)
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]

		classNumStr = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector(trainingPath + r'\%s' % fileNameStr)

	#identify
	vectorUnderTest = image2vector(fileName);
	classifierResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)

	print 'the number is :%d' % classifierResult
	return classifierResult

if __name__ == '__main__':
	identify(sys.argv[1])
