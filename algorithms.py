import csv
import numpy
import matplotlib.pyplot as plt

from config import fileOut, setCV
from scipy import interp
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def fileName(algorithm):
	return "Output/" + fileOut + algorithm + ".csv"

# Analyze model complexity curve for NonBoost tree classifier, max_depth
def nonboostMaxDepth(nonboostDepth, input_list, output_list, test_input, test_output):
	file = open(fileName("_nonboost_max_depth_results"), "w")
	print("Beginning model complexity analysis for NonBoost...")
	file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for max_depth in range(nonboostDepth):
		print("\tNonboost Progress: "+str(int(max_depth/nonboostDepth*100))+"%", end='\r')
		classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
		file.write(str(max_depth + 1) + ","
		+ str(cross_val_score(classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		file.write(str(classifier.score(input_list, output_list)) + ", ")
		file.write(str(classifier.score(test_input, test_output)) + "\n")
	print("\tNonboost Progress: 100%")

# Analyze model complexity curve for AdaBoost tree classifier, find n_estimators
def adaboostNEst(adaboostEstimators, input_list, output_list, test_input, test_output):
	file = open(fileName("_adaboost_n_estimators_results"), "w")
	print("Beginning model complexity analysis for AdaBoost... n_estimators")
	file.write("n_estimators" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for n_estimators in range(adaboostEstimators):
		print("\tAdaboost n-Estimator Progress: "+ str(int(n_estimators/adaboostEstimators*100))+"%", end="\r")
		classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)
		file.write(str((n_estimators + 1) * 10) + "," + str(cross_val_score(
		classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		file.write(str(classifier.score(input_list, output_list)) + ", ")
		file.write(str(classifier.score(test_input, test_output)) + "\n")
	print("\tAdaboost n-Estimator Progress: 100%")

# Analyze model complexity curve for AdaBoost tree classifier, find max_depth
def adaboostMaxDepth(adaboostDepth, input_list, output_list, test_input, test_output):
	file = open(fileName("_adaboost_max_depth_results"), "w")
	print("Beginning model complexity analysis for AdaBoost... max_depth")
	file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for max_depth in range(adaboostDepth):
		print("\tAdaboot Max-Depth Progress: "+str(int(max_depth/adaboostDepth*100))+"%", end="\r")
		classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=50)
		result = ""
		result += (str(max_depth + 1) + "," + str(cross_val_score(
		classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		result += str(classifier.score(input_list, output_list)) + ", "
		result += str(classifier.score(test_input, test_output)) + "\n"
		file.write(result)
	print("\tAdaboost Max-Depth Progress: 100%")

# Analyze model complexity curve for KNN classifier
def knn(knnK, input_list, output_list, test_input, test_output):
	file = open(fileName("_knn_results"), "w")
	print("Beginning model complexity analysis for KNN...")
	file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for k in range(knnK):
		print("\tKNN Progress: " +str(int(k/knnK*100)) + "%", end="\r")
		classifier = KNeighborsClassifier(n_neighbors = k + 1)
		result = ""
		result += (str(k + 1) + "," + str(cross_val_score(
		classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		result += str(classifier.score(input_list, output_list)) + ", "
		result += str(classifier.score(test_input, test_output)) + "\n"
		file.write(result)
	print("\tKNN Progress: 100%")

# Neural network ideal number of neurons in a layer
def ANNVaryNeurons(varyNeurons, input_list, output_list, test_input, test_output):
	scaler = StandardScaler()
	scaler.fit(input_list)
	input_list = scaler.transform(input_list)
	test_input = scaler.transform(test_input)
	file = open(fileName("_neural_network_layer_results"), "w")
	print("Beginning model complexity analysis for NeuralNetwork... neurons")
	file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for neurons in range(varyNeurons):
		print("\tANN Vary Neuron Progress: " +str(int(neurons/varyNeurons*100)) + "%", end="\r")
		layers = [neurons + 1]
		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
		result = ""
		result += (str(neurons + 1) + "," + str(cross_val_score(
		classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		result += str(classifier.score(input_list, output_list)) + ", "
		result += str(classifier.score(test_input, test_output)) + "\n"
		file.write(result)
	print("\tANN Vary Neuron Progress: 100%")

# Neural network tuple length analysis, or number of layers
def ANNVaryLayers(varyLayers, input_list, output_list, test_input, test_output):
	scaler = StandardScaler()
	scaler.fit(input_list)
	input_list = scaler.transform(input_list)
	test_input = scaler.transform(test_input)
	file = open(fileName("_neural_network_layer_length_results"), "w")
	print("Beginning model complexity analysis for NeuralNetwork... number of layers")
	file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
	for numLayer in range(varyLayers):
		print("\tANN Vary Layers Progress: " +str(int(numLayer/varyLayers*100)) + "%", end="\r")
		layers = []
		for neuron in range(numLayer):
		    layers.append(18)
		classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
		result = ""
		result += (str(neurons + 1) + "," + str(cross_val_score(
		classifier, input_list, output_list, cv = setCV).mean()) + ", ")
		classifier.fit(input_list, output_list)
		result += str(classifier.score(input_list, output_list)) + ", "
		result += str(classifier.score(test_input, test_output)) + "\n"
		file.write(result)
	print("\tANN Vary Layers Progress: 100%")
'''
# SVC kernel analysis; which kernel is ideal
SVC = svm.SVC(); # rbf
SigmoidSVC = svm.SVC(kernel="sigmoid")
LinearSVC = svm.LinearSVC();

result = ""
result += ("RBF_SVC" + "," + str(cross_val_score(
SVC, input_list, output_list, cv = setCV).mean()) + ", ")
SVC.fit(input_list, output_list)
result += str(SVC.score(input_list, output_list)) + ", "
result += str(SVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Sigmoid_SVC" + "," + str(cross_val_score(
SigmoidSVC, input_list, output_list, cv = setCV).mean()) + ", ")
SigmoidSVC.fit(input_list, output_list)
result += str(SigmoidSVC.score(input_list, output_list)) + ", "
result += str(SigmoidSVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Linear_SVC" + "," + str(cross_val_score(
LinearSVC, input_list, output_list, cv = setCV).mean()) + ", ")
LinearSVC.fit(input_list, output_list)
result += str(LinearSVC.score(input_list, output_list)) + ", "
result += str(LinearSVC.score(test_input, test_output)) + "\n"
print(result)

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
