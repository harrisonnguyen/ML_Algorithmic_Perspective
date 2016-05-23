import numpy as np

def calc_entropy(p):
	if p!=0:
		return -p*np.log2(p)
	else:
		return 0

def calc_info_gain(data,classes,feature):
	gain = 0
	numData = len(data)
	#find unique values
	values = []
	for datapoint in data:
		if datapoint[feature] not in values:
			values.append(datapoint[feature])
	
	featureCounts = np.zeros(len(values))
	entropy = np.zeros(len(values))
	
	valueIndex = 0
	#find where those values appear in data[feature] and the corresponding class
	for value in values:
		dataIndex = 0
		#array to store the classes of datapoint with the specific feature value
		newClasses = []
		for datapoint in data:
			if datapoint[feature] == value:
				featureCounts[valueIndex]+=1
				newClasses.append(classes[dataIndex])
			dataIndex +=1
			
		#get the unique values in newClasses	
		classValues = set(newClasses)
		
		classCounts = np.zeros(len(classValues))
		classIndex = 0
		for classValue in classValues:
			classCounts[classIndex] = newClasses.count(classValue)
			classIndex +=1
		
		for classIndex in range(len(classValues)):
			entropy[valueIndex] +=calc_entropy(float(classCounts[classIndex])/sum(classCounts))
			
		gain+= float(featureCounts[valueIndex])/numData*entropy[valueIndex]
		valueIndex +=1
		
	return gain
		
	