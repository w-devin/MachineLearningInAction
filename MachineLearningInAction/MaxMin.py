#encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib.animation as animi
import numpy as np
import math
import random
import time

def distance(List, a, b):
	return math.sqrt((List[a][0] - List[b][0]) ** 2 + (List[a][1] - List[b][1]) ** 2)

def MaxMin(Points, Theta, A = 0): 
	Kernals = []
	n = Points.size >> 1
	Distance = np.zeros(n)
	Class = np.array(np.zeros(n))

	#first class kernal
	if A < 0:
		A = random.randint(0, n - 1)
	Kernals += [A]
	Class[A] = 1

	#second class kernal
	for i in range(n):
		Distance[i] = distance(Points, A, i)
	B = np.where(Distance == Distance.max())
	Class[B] = len(Kernals)

	#other kernals
	Dis = Distance[B] 	
	Distance = np.zeros(n) + sys.float_info.max
	while(True):
		x = -1
		for r in set(range(0, n)) - set(Kernals):
			for i in Kernals:
				Distance[r] = min(Distance[r], distance(Points, i, r))
			if x < 0 or Distance[x] < Distance[r]: x = r
 
		if Dis * Theta <= Distance[x]:
			Kernals += [x]
			Class[x] = len(Kernals)
			continue
		break

	#print Kernals
	plt.figure('Result')
	for x in Kernals:
		plt.plot(Points[x, 0], Points[x, 1], 'r+')

	#classify
	for i in set(range(n)) - set(Kernals):
		d = -1; x = -1
		for r in Kernals:
			dd = distance(Points, i, r)
			if d < 0 or d > dd: d = dd; x = r
		Class[i] = Class[x]

	plt.scatter(Points[:, 0], Points[:, 1], 32, Class)
	plt.colorbar()
	plt.show()