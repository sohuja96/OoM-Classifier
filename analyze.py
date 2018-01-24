import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from config import fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows, endText
from itertools import cycle
from scipy import interp
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def fileName(algorithm):
	return outputLocation + algorithm + ".csv"
def orderOfMagnitude(record):
    floatedValue = float(record[categorizingAttributeNumber])
    try:
        return float(math.floor(math.log10(floatedValue)));  #mean
    except ValueError:
        return - 1000000

with open(fileIn, "r", encoding="utf8") as file:
    reader = csv.reader(file)
    csv_list = list(reader)
    csv_list = csv_list[1:numRows] #Ignore first row of data
    csv_list = shuffle(csv_list, n_samples=len(csv_list)) # Randomize

outputLocation = "Output/"+fileOut
setCount = 10
partition_input = []
partition_output = []
input_list = []
output_list = []
test_input = []
test_output = []

for record in csv_list:
    input_vector = []
    try:
    	for attribute in attributeNumbers:
    		buoyantAttribute = float(record[attribute])
    		input_vector.append(buoyantAttribute)
    except ValueError:
        print("ValueError: ", ValueError)
    output_vector = orderOfMagnitude(record)
    partition_input.append(input_vector)
    partition_output.append(output_vector)

input_list = partition_input[:int(0.9 * len(partition_input))]
output_list = partition_output[:int(0.9 * len(partition_input))]
test_input = partition_input[int(0.9 * len(partition_input)):]
test_output = partition_output[int(0.9 * len(partition_input)):]
# Analyze model complexity curve for NonBoost tree classifier, max_depth
file = open(fileName("_nonboost_max_depth_results"), "w")
print("Beginning model complexity analysis for NonBoost...")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(100):
    classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
    file.write(str(max_depth + 1) + ","
    + str(cross_val_score(classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")
'''
# Analyze model complexity curve for AdaBoost tree classifier, find n_estimators
file = open(fileName("_adaboost_n_estimators_results"), "w")
print("Beginning model complexity analysis for AdaBoost... n_estimators")
file.write("n_estimators" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for n_estimators in range(50):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)
    file.write(str((n_estimators + 1) * 10) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")

# Analyze model complexity curve for AdaBoost tree classifier, find max_depth
file = open(fileName("_adaboost_max_depth_results"), "w")
print("Beginning model complexity analysis for AdaBoost... max_depth")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(100):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=50)
    result = ""
    result += (str(max_depth + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)

# Analyze model complexity curve for KNN classifier
file = open(fileName("_knn_results"), "w")
print("Beginning model complexity analysis for KNN...")
file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for k in range(250):
    classifier = KNeighborsClassifier(n_neighbors = k + 1)
    result = ""
    result += (str(k + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    file.write(result)

# Neural network ideal number of neurons in a layer
scaler = StandardScaler()
scaler.fit(input_list)
input_list = scaler.transform(input_list)
test_input = scaler.transform(test_input)

file = open(fileName("_neural_network_layer_results"), "w")
print("Beginning model complexity analysis for NeuralNetwork... neurons")
file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for neurons in range(100):
    layers = [neurons + 1]
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    result = ""
    result += (str(neurons + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)

# Neural network tuple length analysis, or number of layers
scaler = StandardScaler()
scaler.fit(input_list)
input_list = scaler.transform(input_list)
test_input = scaler.transform(test_input)

file = open(fileName("_neural_network_layer_length_results"), "w")
print("Beginning model complexity analysis for NeuralNetwork... number of layers")
file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for neurons in range(100):
    layers = []
    for neuron in range(neurons):
        layers.append(18)
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    result = ""
    result += (str(neurons + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)

# SVC kernel analysis; which kernel is ideal
SVC = svm.SVC(); # rbf
SigmoidSVC = svm.SVC(kernel="sigmoid")
LinearSVC = svm.LinearSVC();

result = ""
result += ("RBF_SVC" + "," + str(cross_val_score(
SVC, input_list, output_list, cv = setCount).mean()) + ", ")
SVC.fit(input_list, output_list)
result += str(SVC.score(input_list, output_list)) + ", "
result += str(SVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Sigmoid_SVC" + "," + str(cross_val_score(
SigmoidSVC, input_list, output_list, cv = setCount).mean()) + ", ")
SigmoidSVC.fit(input_list, output_list)
result += str(SigmoidSVC.score(input_list, output_list)) + ", "
result += str(SigmoidSVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Linear_SVC" + "," + str(cross_val_score(
LinearSVC, input_list, output_list, cv = setCount).mean()) + ", ")
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
    output += str(cross_val_score(DT, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(AB, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(KN, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(N, input_nn_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(SV, input_partition, output_partition, cv = setCount).mean()) + ", "
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

# Messy code below for generating ROC curves and confusion matrices
layers = []
for i in range(14):
    layers.append(18)

file = open(fileName("roc_curve_data_svc"), "w")
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
color = "red"
NB1 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), n_estimators=50)
preds = NB1.fit(input_list, output_list).predict(test_input)
probs = NB1.fit(input_list, output_list).predict_proba(test_input)
cm = confusion_matrix(test_output, preds)
print(cm)
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
print(endText)