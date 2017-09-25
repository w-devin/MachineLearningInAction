#encoding = utf-8
from numpy import *
import operator

#norm
def autoNorm(dataSet):
	minVals = dataSet.min(axis = 0)
	maxVals = dataSet.max(axis = 0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))

	return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]

	#calculate distance [sqrt(sigama:power(2))]
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