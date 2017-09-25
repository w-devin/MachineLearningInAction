#encoding=utf-8
import matplotlib
import matplotlib.pyplot as plot
from numpy import *
import os


#plot the datas
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group[:, 1], group[:, 2])
plt.xlabel(r'The percentage of time spent playing video games')
plt.ylabel(r'Weekly consumption of ice cream')
ax.scatter(group[:, 1], group[:, 2], 15.0 * array(labels), 15.0 * array(labels))
ax.scatter(group[:, 0], group[:, 1], 32 * array(labels), 32 * array(labels))
#plt.show()
'''


import KNN_HandWriting

#Datas = KNN_HandWriting.img2vector(r'Datas/digits/testDigits/0_0.txt')

#print Datas[0, 0:31]

#import KNN_Date

#KNN_HandWriting.handwritingClassTest()


#test with identify the number in image file, but with very low efficienty
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

