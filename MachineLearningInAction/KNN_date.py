#encoding = utf-8
from numpy import *
import operator
import KNN


#	
# 1000 people's percentage of time spent playing video game, frequent flier miles earned per year, liters of ice cream consumed per year and the degree of preference to these guys
# input one's data, output the degree of preference of the one
#

#create test datas
def createDataSet():
	group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
	labels = array(['A', 'A', 'A', 'B', 'B', 'B'])
	return group, labels

#read datas from text file
def file2matrix(filename):
	fr = open(filename)
	arrayofLines = fr.readlines()
	numberofLines = len(arrayofLines)

	returnMat = zeros((numberofLines, 3))
	classLabelVector = []
	index = 0

	for line in arrayofLines:
		line =line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1

	return returnMat, classLabelVector

#test the classify algorithm
def datingClassTest():
	hoRatio= 0.10

	datingDataMat, datingLabels = file2matrix('./Datas/DatingTestSet/datingTestSet2.txt')
	normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0

	for i in range(numTestVecs):
		classifierResult = KNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		print 'the Classifier came back with: %d, the real answer is %d' % (classifierResult, datingLabels[i])
		if(classifierResult != datingLabels[i]): errorCount += 1.0

	print 'the total error rate is : %f' % (errorCount / float(numTestVecs))

def main():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input('percnetage of time spent playing video games?'))
	ffMiles = float(raw_input('frequent flier miles earned per year?'))
	iceCream = float(raw_input('liters of ice cream consumed per year?'))

	datingDataMat, datingLabels = file2matrix('./Datas/DatingTestSet/datingTestSet2.txt')
	normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = KNN.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)

	print 'You will probably like this person:', resultList[classifierResult - 1]

if __name__ == '__main__':
	main()