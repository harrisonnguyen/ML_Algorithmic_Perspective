import numpy as np
class pcn(object):
	def __init__(self,inputs,targets):
		# add colummn of ones to end of data
				# Set up network size
		if np.ndim(inputs)>1:
			self.nIn = np.shape(inputs)[1]
		else: 
			self.nIn = 1
	
		if np.ndim(targets)>1:
			self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(inputs)[0]
	
		# Initialise network
		self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05
	
	def pcnfwd(self,inputs):
		#compute activations
		activations = np.dot(inputs,self.weights)
		
		#threshold the activations
		return np.where(activations>0,1,0)
		
	def pcntrain(self,inputs,targets,eta,T):
		# add the bias node
		inputs = np.concatenate((inputs,-np.ones((inputs.shape[0],1))),axis=1)
		for i in range(0,T):
			self.activations = self.pcnfwd(inputs)
			self.weights -=eta*np.dot(np.transpose(inputs),self.activations-targets)
			
			#print "Iteration: %d" %i
			#print self.weights
			
			activations = self.pcnfwd(inputs)
			#print "Final outputs are:" 
			#print activations
			
	def confmat(self,inputs,targets):
		"""Confusion matrix"""

		# Add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		outputs = np.dot(inputs,self.weights)
	
		nClasses = np.shape(targets)[1]

		if nClasses==1:
			nClasses = 2
			outputs = np.where(outputs>0,1,0)
		else:
			# 1-of-N encoding
			outputs = np.argmax(outputs,1)
			targets = np.argmax(targets,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print cm
		print np.trace(cm)/np.sum(cm)
		
	
	