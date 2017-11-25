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
	#A = random.randint(0, n - 1)
	Kernals += [A]
	Class[A] = 1

	#second class kernal
	B = -1
	for i in range(n):
		Distance[i] = distance(Points, A, i)
		if B < 0 or Distance[B] < Distance[i]: B = i
	Kernals += [B]
	Class[B] = 2

	#other kernals
	Dis = Distance[B]
	Distance = np.zeros(n)
	while(True):
		x = -1
		for r in set(range(0, n)) - set(Kernals):
			for i in Kernals:
				if Distance[r] == 0:
					Distance[r] = distance(Points, i, r)
				else: Distance[r] = min(Distance[r], distance(Points, i, r))
			if x < 0 or Distance[x] < Distance[r]: x = r
 
		if Dis * Theta <= Distance[x]:
			Kernals += [x]
			Class[x] = len(Kernals)
		else:
			break
		if len(Kernals) == n:
			break

	#print Kernals

	plt.figure('Result')
	for x in Kernals:
		plt.plot(Points[x, 0], Points[x, 1], 'r+')
	#plt.scatter(Points[:, 0], Points[:, 1], marker = '+', c = 'r', s = Class * 64)

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