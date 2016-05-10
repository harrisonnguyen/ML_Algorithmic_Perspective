import numpy as np
from scipy import linalg as la

def pca(data,redDim=0,normalise =1):
	#centre the data by subtracting the mean of each column
	mu = np.mean(data,axis=0)
	data -=mu
	
	#compute the covariance matrix
	C = np.cov(np.transpose(data))
	evals,evecs = la.eig(C)
	
	#sort the eigenvals from descending order
	indices = np.argsort(evals)
	indices = indices[::-1]
	evecs = evecs[:,indices]
	evals = evals[indices]
	
	#we get the top redDim eigenvectors
	if redDim> 0:
		evecs = evecs[:,:redDim]
	if normalise:
		for i in range(np.shape(evecs)[1]):
			evecs[:,i] / np.linalg.norm(evecs[:,i]) * np.sqrt(evals[i])
	
	#construct the new data matrix
	x = np.dot(np.transpose(evecs),np.transpose(data))
	
	y = np.transpose(np.dot(evecs,x))+mu
	
	return x,y,evecs,evals
	