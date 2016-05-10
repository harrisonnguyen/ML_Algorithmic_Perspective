import numpy as np
from scipy import linalg as la

def ld_analysis(data,labels,redDim):
	nData = np.shape(data)[0]
	C = np.cov(np.transpose(data))
	classes = np.unique(labels)
	Sw = 0
	for i in range(len(classes)):
		indices = np.squeeze(np.where(labels == classes[i])) #get the indces of data with label i
		d = np.squeeze(data[indices,:]) #get get of label i
		classcov = np.cov(np.transpose(d)) #compute covariance of the class
		Sw += np.float(np.shape(indices)[0])/nData*classcov # np.float(np.shape(indices)[0])/nData = pc
		
	Sb = C- Sw
	evals,evecs = la.eig(Sw,Sb)
	indices = np.argsort(evals) #higher values, more variance, more important in explaining
	indices = indices[::-1] #reverse the direciton
	evecs = evecs[:,indices]
	evals = evals[indices]
	w = evecs[:,:redDim]
	newData = np.dot(data,w)
	return newData