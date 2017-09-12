#encoding=utf-8
import KNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

datingDataMat, labels = KNN.file2matrix('datingTestSet2.txt')
normMat, ranges, minVales = KNN.autoNorm(datingDataMat)

print normMat
print ranges
print minVales

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(group[:, 1], group[:, 2])
#plt.xlabel(r'The percentage of time spent playing video games')
#plt.ylabel(r'Weekly consumption of ice cream')
#ax.scatter(group[:, 1], group[:, 2], 15.0 * array(labels), 15.0 * array(labels))
#ax.scatter(group[:, 0], group[:, 1], 32 * array(labels), 32 * array(labels))

#plt.show()