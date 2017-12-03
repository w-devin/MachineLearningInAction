import numpy as np

def LinerClassify(X, Class, W, rho = 0.03):
	X = np.concatenate((X, np.ones((4, 1))), axis = 1)
	for i in range(len(X)):
		if Class[i] == 1: X[i] *= -1	
	X = np.mat(X)
	W = W.reshape(len(W), 1)

	k = 0; i = 0
	while(True):
		k += 1
		i = (i + 1) % len(X)
		dx = X[i] * W
		
		if dx <= 0:
			W = np.array([W.tolist(), (X[i] * rho).reshape(len(W), 1).tolist()]).sum(axis = 0)
			k = 0
			print 'w -> ', W.reshape(1, len(W))

		if k == len(X): return W