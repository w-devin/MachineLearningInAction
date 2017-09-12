#encoding = utf-8
from numpy import *
import operator

def createDataSet():
	group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
	labels = array(['A', 'A', 'A', 'B', 'B', 'B'])
	return group, labels

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


def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]

	#calculate distance
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances ** 0.5

	#Select the nearest two points
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

	return sortedClassCount[0][0]

#norm
def autoNorm(dataSet):
	minVals = dataSet.min(axis = 0)
	print 'minvals = ' , minVals
	maxVals = dataSet.max(axis = 0)
	print 'maxvals = ' , maxVals
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))


	return normDataSet, ranges, minVals




















