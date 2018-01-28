import argparse

# Parse CLI
parser = argparse.ArgumentParser()
parser.add_argument("-nbd", "--nonboostDepth", help="Nonboosted Decision Tree:\tMax Depth", type=int)
parser.add_argument("-abe", "--adaboostEstimators", help="Boosted Decision Tree:\tn n-Estimators", type=int)
parser.add_argument("-abd", "--adaboostDepth", help="Boosted Decision Tree:\t Max Depth", type=int)
parser.add_argument("-knn", "--knnK", help="K-Nearest Neighbors:\tVary K", type=int)
parser.add_argument("-vn", "--varyNeurons", help="Artificial Neural Network:\tVary Neurons", type=int)
parser.add_argument("-vl", "--varyLayers", help="Artificial Neural Network:\tVary Layers", type=int)
parser.add_argument("-svm", "--supportVectorMachines", help="Support Vector Machines:\tVary Kernels", action="store_true")
arguments = parser.parse_args()

print("\nStart of Machine Learning")

import algorithms2
import warnings
from config2 import fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows

warnings.simplefilter("ignore")

# Output the set-up parameters
parameters = [fileIn, fileOut, attributeNumbers, categorizingAttributeNumber, numRows,
arguments.nonboostDepth, arguments.adaboostEstimators, arguments.adaboostDepth,
arguments.knnK, arguments.varyNeurons, arguments.varyLayers]
parameterNames = ["fileIn", "fileOut", "attributeNumbers", "categorizingAttributeNumber", "numRows", 
"nonboostDepth", "adaboostEstimators", "adaboostDepth", "knnK", "varyNeurons", "varyLayers"]
parameterInfo = ""
for param in range(0, len(parameters)):
	parameterInfo += "    " + parameterNames[param] + "=" + str(parameters[param]) + "\n"
print("Running analysis with configuration:\n" + parameterInfo) 

# Run the selected algorithms
if not arguments.nonboostDepth == None:
	algorithms2.nonboostMaxDepth(arguments.nonboostDepth)
if not arguments.adaboostEstimators == None:
	algorithms2.adaboostNEst(arguments.adaboostEstimators)
if not arguments.adaboostDepth == None:
	algorithms2.adaboostMaxDepth(arguments.adaboostDepth)
if not arguments.knnK == None:
	algorithms2.knn(arguments.knnK)
if not arguments.varyNeurons == None:
	algorithms2.ANNVaryNeurons(arguments.varyNeurons)
if not arguments.varyLayers == None:
	algorithms2.ANNVaryLayers(arguments.varyLayers)
if arguments.supportVectorMachines:
	algorithms2.svmCompare()

print("\nMachine Learning Complete!\n")