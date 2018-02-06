import csv
import math
import numpy
import matplotlib.pyplot as plt

from config import fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows, setCV
from scipy import interp
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Output consistent file names
def fileName(algorithm):
	return "Output/" + fileOut + algorithm + ".csv"

# Calculate order of magnitude
def orderOfMagnitude(record):
    floatedValue = float(record[categorizingAttributeNumber-1])
    try:
        return float(math.floor(math.log10(floatedValue)));  #mean
    except ValueError:
        return - 1000000

# Open the dataset, format and shuffle it
with open(fileIn, "r", encoding="utf8") as file:
    reader = csv.reader(file)
    csv_list = list(reader)
    csv_list = csv_list[1:] #Ignore first row of data
    csv_list = shuffle(csv_list, n_samples=len(csv_list)) # Randomize
    csv_list = csv_list[:numRows] 

# Set up data containers
partition_input = []
partition_output = []
test_input = []
test_output = []

# Move dataset into input and outputs
errors= 0
inputs= 0

for record in csv_list:
	inputs += 1
	input_vector = []
	try:
		for attribute in attributeNumbers:
			attribute -= 1
			buoyantAttribute = float(record[attribute])
			input_vector.append(buoyantAttribute)
	except ValueError:
		errors += 1
		print("ValueError for " + str(record[attribute]) + "\n\t" + str(errors/inputs *100) + "% errors in your dataset")
	output_vector = orderOfMagnitude(record)
	partition_input.append(input_vector)
	partition_output.append(output_vector)

# Make training and testing data
input_list = partition_input[:int(0.9 * len(partition_input))]
output_list = partition_output[:int(0.9 * len(partition_input))]
test_input = partition_input[int(0.9 * len(partition_input)):]
test_output = partition_output[int(0.9 * len(partition_input)):]

# Analyze model complexity curve for NonBoost tree classifier, max_depth
def nonboostMaxLeafNodes(nonboostDepth):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_nonboost_max_leaf_nodes_results"), "w")
	file.write("max_depth" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)

	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for NonBoost...Max Leaf")
	for max_depth in range(nonboostDepth):
		print("\t  Nonboost Progress: "
		+str(int(max_depth/nonboostDepth*100))
		+"%", end='\r')

		classifier = tree.DecisionTreeClassifier(max_leaf_nodes = max_depth + 2)

		depths.append(max_depth + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str(max_depth + 1) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")

	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Max Leaves', ylabel='Percent Correct', title='Nonboost Max Leaf Node CV')
	testAxis.set(xlabel='Max Leaves', ylabel='Percent Correct', title='Nonboost Max Leaf Node Testing')
	figure.savefig(fileName("nonboost_leaf")+".png")

	print("\t  Nonboost Progress: 100%")
# Analyze model complexity curve for NonBoost tree classifier, max_depth
def nonboostMaxDepth(nonboostDepth):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_nonboost_max_depth_results"), "w")
	file.write("max_depth" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)

	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for NonBoost...")
	for max_depth in range(nonboostDepth):
		print("\t  Nonboost Progress: "
		+str(int(max_depth/nonboostDepth*100))
		+"%", end='\r')

		classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)

		depths.append(max_depth + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str(max_depth + 1) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")

	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Max Depth', ylabel='Percent Correct', title='Nonboost Max Depth CV')
	testAxis.set(xlabel='Max Depth', ylabel='Percent Correct', title='Nonboost Max Depth Testing')
	figure.savefig(fileName("nonboost")+".png")

	print("\t  Nonboost Progress: 100%")

# Analyze model complexity curve for AdaBoost tree classifier, find n_estimators
def adaboostNEst(adaboostEstimators):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_adaboost_n_estimators_results"), "w")
	file.write("n_estimators" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for AdaBoost... n_estimators")
	for n_estimators in range(adaboostEstimators):
		print("\t  Adaboost n-Estimator Progress: "+ str(int(n_estimators/adaboostEstimators*100))+"%", end="\r")

		classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)

		depths.append(n_estimators + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str((n_estimators + 1) * 10) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")

	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='n-Estimator', ylabel='Percent Correct', title='Adaboost n-Estimator CV')
	testAxis.set(xlabel='n-Estimator', ylabel='Percent Correct', title='Adaboost n-Estimator Testing')
	figure.savefig(fileName("adaboost_n_estimator")+".png")

	print("\t  Adaboost n-Estimator Progress: 100%")

# Analyze model complexity curve for AdaBoost tree classifier, find max_depth
def adaboostMaxDepth(adaboostDepth):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_adaboost_max_depth_results"), "w")
	file.write("max_depth" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for AdaBoost... max_depth")
	for max_depth in range(adaboostDepth):
		print("\t  Adaboot Max-Depth Progress: "+str(int(max_depth/adaboostDepth*100))+"%", end="\r")

		classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=50)

		depths.append(max_depth + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str((max_depth + 1) * 10) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Max Depth', ylabel='Percent Correct', title='Adaboost Max Depth CV')
	testAxis.set(xlabel='Max Depth', ylabel='Percent Correct', title='Adaboost Max Depth Testing')
	figure.savefig(fileName("adaboost_max_depth")+".png")

	print("\t  Adaboost Max-Depth Progress: 100%")

# Analyze model complexity curve for KNN classifier
def knn(knnK):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_knn_results"), "w")
	file.write("k" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for KNN...")
	for k in range(knnK):
		print("\t  KNN Progress: " +str(int(k/knnK*100)) + "%", end="\r")

		classifier = KNeighborsClassifier(n_neighbors = k + 1)

		depths.append(k + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str((k + 1) * 10) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='k', ylabel='Percent Correct', title='KNN CV')
	testAxis.set(xlabel='k', ylabel='Percent Correct', title='KNN Testing')
	figure.savefig(fileName("knn")+".png")

	print("\t  KNN Progress: 100%")

# Analyze model complexity curve for KNN classifier
def knnDistance(knnK):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_knn_distance_results"), "w")
	file.write("k" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for KNN...Distance")
	for k in range(knnK):
		print("\t  KNN Progress: " +str(int(k/knnK*100)) + "%", end="\r")

		classifier = KNeighborsClassifier(n_neighbors = k + 1,
                weights='distance')

		depths.append(k + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str((k + 1) * 10) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='k', ylabel='Percent Correct', title='KNN CV')
	testAxis.set(xlabel='k', ylabel='Percent Correct', title='KNN Testing')
	figure.savefig(fileName("knn_distance")+".png")

	print("\t  KNN Progress: 100%")


# Neural network ideal number of neurons in a layer
def ANNVaryNeurons(varyNeurons):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_neural_network_layer_results"), "w")
	file.write("layers" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	scaler = StandardScaler()
	scaler.fit(input_list)
	input_list = scaler.transform(input_list)
	test_input = scaler.transform(test_input)

	print("\tBeginning model complexity analysis for NeuralNetwork... neurons")
	for neurons in range(varyNeurons):
		print("\t  ANN Vary Neuron Progress: " +str(int(neurons/varyNeurons*100)) + "%", end="\r")
		layerSize = [neurons + 1]
		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layerSize), random_state=1)

		depths.append(neurons + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str(neurons + 1) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Neurons', ylabel='Percent Correct', title='ANN CV')
	testAxis.set(xlabel='Neurons', ylabel='Percent Correct', title='ANN Testing')
	figure.savefig(fileName("ANNVaryNeurons")+".png")
	print("\t  ANN Vary Neuron Progress: 100%")

# Neural network tuple length analysis, or number of layers
def ANNVaryLayers(varyLayers):
	global input_list
	global output_list
	global test_input
	global test_output

	scaler = StandardScaler()
	scaler.fit(input_list)
	input_list = scaler.transform(input_list)
	test_input = scaler.transform(test_input)
	file = open(fileName("_neural_network_layer_length_results"), "w")
	file.write("layers" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	scaler = StandardScaler()
	scaler.fit(input_list)
	input_list = scaler.transform(input_list)
	test_input = scaler.transform(test_input)

	print("\tBeginning model complexity analysis for NeuralNetwork... number of layers")
	for numLayer in range(varyLayers):
		print("\t  ANN Vary Layers Progress: " +str(int(numLayer/varyLayers*100)) + "%", end="\r")
		layers = []
		for neuron in range(numLayer):
		    layers.append(18)
		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
		depths.append(numLayer + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str(numLayer + 1) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Layers', ylabel='Percent Correct', title='ANN CV')
	testAxis.set(xlabel='Layers', ylabel='Percent Correct', title='ANN Testing')
	figure.savefig(fileName("ANNVaryLayers")+".png")
	print("\t  ANN Vary Layers Progress: 100%")

# SVC kernel analysis; which kernel is ideal
def svmCompare():
	print("\tBeginning model complexity analysis for SVM...")
	file = open(fileName("_SVM_results"), "w")
	file.write("layers" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	axis = figure.add_subplot(111)
	barWidth = 0.35
	opacity = 1
	index = numpy.arange(3) #num Kernels
	cvScores = []
	testScores = []

	print("\t  Radial Basis Function")
	SVC = svm.SVC() # rbf

	crossValScore = cross_val_score(SVC, input_list, output_list, cv = setCV).mean()
	SVC.fit(input_list, output_list)
	testScore = SVC.score(test_input, test_output)

	cvScores.append(crossValScore)
	testScores.append(testScore)

	result = ""
	result += ("RBF_SVC"
	+ ","
	+ str(crossValScore)
	+ ", ")
	result += str(testScore) + "\n"
	print("\t    Cross-Validation Score: " + str(int(crossValScore*100)) + "%")
	print("\t    Testing Score: " + str(int(testScore*100)) + "%")

	print("\t  Sigmoid")
	SigmoidSVC = svm.SVC(kernel="sigmoid")
	crossValScore = cross_val_score(SigmoidSVC, input_list, output_list, cv = setCV).mean()
	SigmoidSVC.fit(input_list, output_list)
	testScore = SigmoidSVC.score(test_input, test_output)

	cvScores.append(crossValScore)
	testScores.append(testScore)

	result = ""
	result += ("Sigmoid_SVC"
	+ ","
	+ str(crossValScore)
	+ ", ")
	result += str(testScore) + "\n"
	print("\t    Cross-Validation Score: " + str(int(crossValScore*100)) + "%")
	print("\t    Testing Score: " + str(int(testScore*100)) + "%")

	print("\t  Linear")
	LinearSVC = svm.LinearSVC();
	crossValScore = cross_val_score(LinearSVC, input_list, output_list, cv = setCV).mean()
	LinearSVC.fit(input_list, output_list)
	testScore = SVC.score(test_input, test_output)

	cvScores.append(crossValScore)
	testScores.append(testScore)

	result = ""
	result += ("Linear_SVC"
	+ ","
	+ str(crossValScore)
	+ ", ")
	result += str(testScore) + "\n"
	print("\t    Cross-Validation Score: " + str(int(crossValScore*100)) + "%")
	print("\t    Testing Score: " + str(int(testScore*100)) + "%")

	rectCV = axis.bar(index, cvScores, barWidth, color='b', label='CV')
	rectTest = axis.bar(index+barWidth, testScores, barWidth, color='y', label='Test')
	axis.set_ylabel('Percent Correct')
	axis.set_title('Comparison of SVM Kernels')
	axis.set_xticks(index + barWidth / 2)
	axis.set_xticklabels(('RBF','Sigmoid','Linear'))
	axis.legend()
	figure.savefig(fileName("svm kernels")+".png")

def svmMaxIterations(knnK):
	global input_list
	global output_list
	global test_input
	global test_output

	file = open(fileName("_svm_iter_results"), "w")
	file.write("k" + ", " + "cross_val_score" + ", " + "testing_score\n")
	figure = plt.figure()
	cvAxis = figure.add_subplot(111)
	testAxis = figure.add_subplot(111)


	depths = []
	cvScore = []
	testScore = []

	print("\tBeginning model complexity analysis for SVM...Iterations")
	for k in range(knnK):
		k = k * 20
		print("\t  SVM Progress: " +str(int(k/knnK*5)) + "%", end="\r")

		classifier = svm.LinearSVC(max_iter = k)

		depths.append(k + 1)
		cvCurrent = cross_val_score(classifier, input_list, output_list, cv = setCV).mean()
		cvScore.append(cvCurrent)

		classifier.fit(input_list, output_list)

		testCurrent = classifier.score(test_input, test_output)
		testScore.append(testCurrent)

		file.write(str(k + 1) + "," + str(cvCurrent) + ", " + str(testCurrent) + "\n")
	cvAxis.plot(depths, cvScore)
	testAxis.plot(depths, testScore)

	cvAxis.set(xlabel='Max Iterations', ylabel='Percent Correct', title='SVM Iteration CV')
	testAxis.set(xlabel='Max Iterations', ylabel='Percent Correct', title='SVM Iteration Testing')
	figure.savefig(fileName("svm_iter")+".png")

	print("\t  SVM Progress: 100%")
