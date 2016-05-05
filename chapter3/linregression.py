import numpy as np

class linreg_model(object):
	
	def __init__(self,inputs,targets):
		#
		pass
	
	def linreg(self,inputs,targets):
		inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
		beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),
						np.transpose(inputs)),targets)
		return beta
		
	def predict(self,inputs,weights):
		#add column of ones to the data
		inputs = np.concatenate((inputs,-np.ones((inputs.shape[0],1))),axis=1)
		return np.dot(inputs,weights)