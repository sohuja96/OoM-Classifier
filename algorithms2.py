import csv
import math
import numpy
import matplotlib.pyplot as plt

from config2 import fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows, setCV
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
    floatedValue = float(record[int(categorizingAttributeNumber)-1])
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
	try:
		output_vector = orderOfMagnitude(record)
	except ValueError:
		print("Weird Class!")
	partition_input.append(input_vector)
	partition_output.append(output_vector)

# Make training and testing data
input_list = partition_input[:int(0.9 * len(partition_input))]
output_list = partition_output[:int(0.9 * len(partition_input))]
test_input = partition_input[int(0.9 * len(partition_input)):]
test_output = partition_output[int(0.9 * len(partition_input)):]

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
	cvAxis.legend()
	testAxis.legend()
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
	cvAxis.legend()
	testAxis.legend()
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
	cvAxis.legend()
	testAxis.legend()
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
	cvAxis.legend()
	testAxis.legend()
	figure.savefig(fileName("knn")+".png")

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
	cvAxis.legend()
	testAxis.legend()
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
	cvAxis.legend()
	testAxis.legend()
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

'''
# Gather data for learning curves
scaler = StandardScaler()

layers = []
for i in range(14):
    layers.append(18)

file = open(fileName("_learning_curve_data.csv"), "w")
print("Beginning learning curve analysis...")
file.write("input_size" + ", " + "cv_dt" + ", " + "cv_ab" + ", " + "cv_kn" + ", " + "cv_n" + ", " + "cv_svc" + ", " + "dt" + ", " + "ab" + ", " + "kn" + ", " + "n" + ", " + "svc\n")
for input_size in range(1, int(len(input_list) / 100)):
    input_partition = input_list[:input_size * 100]
    input_nn_partition = input_list[:input_size * 100]
    scaler.fit(input_nn_partition)
    input_nn_partition = scaler.transform(input_nn_partition)
    output_partition = output_list[:input_size * 100]
    output = str(input_size * 100) + ", "
    DT = tree.DecisionTreeClassifier(max_depth = 6)
    AB = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), n_estimators=50)
    KN = KNeighborsClassifier(n_neighbors = 20)
    N = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    SV = svm.SVC(kernel="sigmoid")
    output += str(cross_val_score(DT, input_partition, output_partition, cv = setCV).mean()) + ", "
    output += str(cross_val_score(AB, input_partition, output_partition, cv = setCV).mean()) + ", "
    output += str(cross_val_score(KN, input_partition, output_partition, cv = setCV).mean()) + ", "
    output += str(cross_val_score(N, input_nn_partition, output_partition, cv = setCV).mean()) + ", "
    output += str(cross_val_score(SV, input_partition, output_partition, cv = setCV).mean()) + ", "
    DT.fit(input_partition, output_partition)
    AB.fit(input_partition, output_partition)
    KN.fit(input_partition, output_partition)
    N.fit(input_nn_partition, output_partition)
    SV.fit(input_partition, output_partition)
    output += str(DT.score(input_partition, output_partition)) + ", "
    output += str(AB.score(input_partition, output_partition)) + ", "
    output += str(KN.score(input_partition, output_partition)) + ", "
    output += str(N.score(input_nn_partition, output_partition)) + ", "
    output += str(SV.score(input_partition, output_partition)) + "\n"
    print(output)
    file.write(output)

# Generating ROC curves and confusion matrices
layers = []
for i in range(14):
    layers.append(18)

file = open(fileName("roc_curve_data_svc"), "w")
mean_tpr = 0.0
mean_fpr = numpy.linspace(0, 1, 100)
lw = 2
color = "red"
fpr, tpr, thresholds = roc_curve(test_output, probs[:, 1])
file.write("tpr" + ", " + "fpr\n")
for i in range(len(fpr)):
    file.write(str(tpr[i]) + ", " + str(fpr[i]) + "\n")
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold (area = %0.2f)' % (roc_auc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('TPR vs. FPR (Coding Survey)')
plt.legend(loc="lower right")
plt.show()
'''

