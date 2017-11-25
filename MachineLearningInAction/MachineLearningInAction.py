#encoding=utf-8
import matplotlib
from numpy import *
import os


#plot the datas
'''
import KNN_Date;
import matplotlib.pyplot as plot
group, labels = KNN_Date.file2matrix(r'Datas/DatingTestSet/datingTestSet2.txt')
fig = plot.figure()
ax = fig.add_subplot(111)
ax.scatter(group[:, 1], group[:, 2])
plot.xlabel(r'The percentage of time spent playing video games')
plot.ylabel(r'Weekly consumption of ice cream')
#ax.scatter(group[:, 1], group[:, 2], 15.0 * array(labels), 15.0 * array(labels))
ax.scatter(group[:, 0], group[:, 1], 32 * array(labels), 32 * array(labels))
plot.show()
'''


#test of the class KNN_HandWriting
'''
import KNN_HandWriting
Datas = KNN_HandWriting.img2vector(r'Datas/digits/testDigits/0_0.txt')
print Datas[0, 0:31]
import KNN_Date
KNN_HandWriting.handwritingClassTest()
'''

#test with identify the number in image file, but with very low efficienty
'''
import KNN_HandWriting
imagePath = r'Datas\digits\images'

testFileList = os.listdir(imagePath)
errorCount = 0.0
mTest = len(testFileList)

for i in range(mTest):
	fileNameStr = testFileList[i]
	fileStr = fileNameStr.split('.')[0]
	classNumStr = int(fileStr.split('_')[0])

	imageFileStr = imagePath + '\\' + fileNameStr

	result = KNN_HandWriting.identify(imageFileStr)

	print '\nthe classifierResult came back with: %d, the real answer is :%d\n' %(result, classNumStr)

	if(result != classNumStr): errorCount += 1

print 'the total number of errors is %d\n' % errorCount
print 'the total error rate is %f\n' % (errorCount / float(mTest))
'''


#Bayes

import NaiveBayes as bayes 
import NaiveBayes_spam as spam
import NaiveBayes_RSS as rss
import feedparser

#test of trainNaiveBayes
'''
listOPosts, listClasses = bayes.loadDataSet()			#generate test data
myVocabList = bayes.createVocabList(listOPosts)			#generate vocabulary list
myVocabList = sort(myVocabList)			#sort the vocabulary list

trainMat = []
for postinDoc in listOPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

p0v, p1v, pAb = bayes.trainNaiveBayes0(trainMat, listClasses)

print pAb
print p0v
print p1v
'''

#test of naive bayes
'''
bayes.testingNaiveBayse()
'''

#test of bagOfWord2Vec
'''
listOPosts, listClasses = bayes.loadDataSet()	
myVocabList = sort(bayes.createVocabList(listOPosts))			
doc = bayes.bagOfWords2VecMN(myVocabList, listOPosts[0])

print doc
'''

#spam mail
'''
import NaiveBayes_spam as spam

spam.spamTest()
'''


#NaiveBayes RSS
'''
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = rss.localWords(ny, sf)
vocabList, pSF, pNY = rss.localWords(ny, sf)
vocabList, pSF, pNY = rss.localWords(ny, sf)
'''

#MaxMin classifier
'''
import MaxMin as mm
import random
import numpy as np

Points = np.array([[0, 0], [3, 8], [2, 2], [1, 1], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]])

mm.MaxMin(Points, 0.6, 0)
#mm.MaxMin(Points, 0.6, random.randint(0, len(Points) - 1))
'''