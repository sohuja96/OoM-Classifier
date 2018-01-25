import algorithms
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

def orderOfMagnitude(record):
    floatedValue = float(record[categorizingAttributeNumber])
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
input_list = []
output_list = []
test_input = []
test_output = []

# Move dataset into input and outputs
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

# Make training and testing data
input_list = partition_input[:int(0.9 * len(partition_input))]
output_list = partition_output[:int(0.9 * len(partition_input))]
test_input = partition_input[int(0.9 * len(partition_input)):]
test_output = partition_output[int(0.9 * len(partition_input)):]

# Parse CLI
parser = argparse.ArgumentParser()
parser.add_argument("-nbd", "--nonboostDepth", help="Nonboosted Decision Tree:\tMax Depth", type=int)
parser.add_argument("-abe", "--adaboostEstimators", help="Boosted Decision Tree:\tn n-Estimators", type=int)
parser.add_argument("-abd", "--adaboostDepth", help="Boosted Decision Tree:\t Max Depth", type=int)
parser.add_argument("-knn", "--knnK", help="K-Nearest Neighbors:\tVary K", type=int)
parser.add_argument("-vn", "--varyNeurons", help="Artificial Neural Network:\tVary Neurons", type=int)
parser.add_argument("-vl", "--varyLayers", help="Artificial Neural Network:\tVary Layers", type=int)
arguments = parser.parse_args()

# Output the set-up parameters
parameters = [fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows,
arguments.nonboostDepth, arguments.adaboostEstimators, arguments.adaboostDepth,
arguments.knnK, arguments.varyNeurons, arguments.varyLayers]
parameterNames = ["fileIn", "fileOut", "attributeNumbers", "categorizingAttributeNumber", "numRows", 
"nonboostDepth", "adaboostEstimators", "adaboostDepth", "knnK", "varyNeurons", "varyLayers"]
parameterInfo = ""
for param in range(0, len(parameters)):
	parameterInfo += "\t" + parameterNames[param] + "=" + str(parameters[param]) + "\n"
print("Running analysis with configuration:\n" + parameterInfo) 
warnings.simplefilter("ignore")

# Run the selected algorithms
if not arguments.nonboostDepth == None:
	algorithms.nonboostMaxDepth(arguments.nonboostDepth, input_list, output_list, test_input, test_output)
if not arguments.adaboostEstimators == None:
	algorithms.adaboostNEst(arguments.adaboostEstimators, input_list, output_list, test_input, test_output)
if not arguments.adaboostDepth == None:
	algorithms.adaboostMaxDepth(arguments.adaboostDepth, input_list, output_list, test_input, test_output)
if not arguments.knnK == None:
	algorithms.knn(arguments.knnK, input_list, output_list, test_input, test_output)
if not arguments.varyNeurons == None:
	algorithms.ANNVaryNeurons(arguments.varyNeurons, input_list, output_list, test_input, test_output)
if not arguments.varyLayers == None:
	algorithms.ANNVaryLayers(arguments.varyLayers, input_list, output_list, test_input, test_output)

print(endText)