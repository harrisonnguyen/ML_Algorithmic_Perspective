import numpy as np

def calc_entropy(p):
	if p!=0:
		return -p*np.log2(p)
	else:
		return 0

def calc_info_gain(data,classes,feature_name):
	gain = 0
	numData = len(data)
	#find unique values
	values = data[feature_name].unique()
	for value in values:
		entropy = 0
		filtered_classes = classes[data[feature_name] == value]
		n_with_value = len(filtered_classes)
		classValues = filtered_classes.unique()
		for classValue in classValues:
			entropy += calc_entropy(float(sum(filtered_classes == classValue))/n_with_value)
		gain += float(n_with_value)/numData*entropy
	return gain

def create_node(feature,isLeaf,left_child,right_child,classes):
	if isLeaf:
		#print classes
		if sum(classes==1) >= sum(classes==0):
			output = 1
		else:
			output = 0
		node_data = {
			'isLeaf': True,
			'output': output
		}
	else:
		node_data = {
			'feature': feature,
			'isLeaf': False,
			'leftChild': left_child,
			'rightChild': right_child,
		}
	return node_data
	
#we build a binary tree. Ensure that the data is in a binary format	
def create_decision_tree(data,classes,featureList,max_depth = 3):
	current_depth = 0
	#create the root and start building the tree
	root = build_tree(data,classes,featureList,current_depth,max_depth)
	
	return root
	


#we build a binary tree. Ensure that the data is in a binary format
def build_tree(data,classes,featureList, current_depth, max_depth):
	#current_depth = 0
	#terminating conditions of the tree
	print classes
	print data
	#if len(data) == 0:
	#	print 'Run out of data'
	#	return None
	if len(featureList) ==0:
		print 'Out of features'
		return create_node(None,True,None,None,classes)
	if sum(classes == classes[0]) == len(classes):
		print 'All classes are in one'
		return create_node(None,True,None,None,classes) #they're all the same classes, so no need to split any further
	if current_depth >= max_depth:
		print 'Reached max depth'
		return create_node(None,True,None,None,classes) #we've reached the maximum depth
	

	
	#find the best feature to split on
	best_gain = -1
	for f in featureList:
		gain = calc_info_gain(data,classes,f)
		if gain >= best_gain:
			best_gain = gain
			best_feature = f
	
	#remove that feature from the list
	tempList = list(featureList)
	tempList.remove(best_feature)
	#featureList.remove(best_feature)
	#split the data and classes according to that feature

	#split the data to right and left sides
	left_logical = data[best_feature] == 0
	left_data_split = data[left_logical]
	left_classes_split = classes[left_logical]
	
	right_logical = data[best_feature] == 1
	right_data_split = data[right_logical]
	right_classes_split = classes[right_logical]
	
	#if data there is no data on either side, we might as well create a leaf
	if len(right_data_split) == 0 or len(left_data_split) == 0:
		print 'Run out of data'
		return create_node(None,True,None,None,classes) #we've reached the maximum depth
		
	#create the left and right children
	left_child = build_tree(left_data_split,left_classes_split,tempList,current_depth+1,max_depth)
	right_child = build_tree(right_data_split,right_classes_split,tempList,current_depth+1,max_depth)
	
	return create_node(best_feature,False,left_child,right_child,None)
	
		
	
	
	

		
	
	
		
	