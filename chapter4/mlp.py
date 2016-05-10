import numpy as np
class mlp(object):
	def __init__(self,inputs,outputs,nhidden,beta=1):
		
		self.nin = np.shape(inputs)[1] #no of input nodes
		self.nout = np.shape(outputs)[1] #no of output nodes
		self.ndata = np.shape(inputs)[0] #no of data points
		self.nhidden = nhidden #no of hidden nodes
		
		self.beta = beta
		
		#initialise the weights of the 3 layer network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
		
	def mlpfwd(self,inputs):
		#compute the activation of each neuon in the hidden layer
		self.hidden = np.dot(inputs,self.weights1)
		self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
		
		#add bias unit to the hidden layer
		self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)
		
		#compute the output later
		outputs = np.dot(self.hidden,self.weights2)
		outputs = 1.0/(1.0 + np.exp(-self.beta*outputs))
		return outputs
		
	def mlptrain(self,inputs,targets,eta,niterations):
		#add bias node to data
		inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
		
		for i in range(0,niterations):
			self.outputs = self.mlpfwd(inputs) #perform forwars phase
			
			#compute the error
			error = np.sum((self.outputs-targets)**2)
			if i %1000 == 0:
				print "Iteration: %d Error: %f" %(i,error)
			delta0 = (self.outputs-targets)*self.outputs*(1-self.outputs) #compute the error at the output
			
			deltah = self.hidden*(1-self.hidden)*np.dot(delta0,np.transpose(self.weights2))
			
			self.weights2 -= eta*np.dot(np.transpose(self.hidden),delta0)
			self.weights1 -= eta*np.dot(np.transpose(inputs),deltah[:,:-1])

	def confmat(self,inputs,targets):
		"""Confusion matrix"""

		# Add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
		outputs = self.mlpfwd(inputs)

		nclasses = np.shape(targets)[1]

		if nclasses==1:
			nclasses = 2
			outputs = np.where(outputs>0.5,1,0)
		else:
			# 1-of-N encoding
			outputs = np.argmax(outputs,1)
			targets = np.argmax(targets,1)

		cm = np.zeros((nclasses,nclasses))
		for i in range(nclasses):
			for j in range(nclasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print "Confusion matrix is:"
		print cm
		print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100			
			