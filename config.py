
fileIn= "" #input file name and location as csv such as "TornadoData/2016_torn.csv"
fileOut= "" #output file name, automatically places it in Output and appends the name of the analysis and .csv file type"
attributeNumbers= [] #column numbers indexing at 1 as [x, y, z, ...], the last one being the attribute you're testing for
categorizingAttributeNumber= 0 #replace with the attribute number (indexing at 1) that you're using as categories
numRows= 10
setCV = 10

nonboostDepth= 50
adaboostDepth= 50
adaboostEstimators= 50
knnK= 50
varyNeurons= 50
varyLayers= 50

# FOR MY LOVELY GTA's! :)
# DATASET 1
fileIn= "TornadoData/2016_torn.csv"
fileOut= "tornado_"
attributeNumbers= [3, 11, 20, 21]
categorizingAttributeNumber= 14
numRows= 1000
# DATASET 2
'''
fileIn= "AirQData/epa_air_quality_annual_summary.csv"
fileOut= "AirQ_"
attributeNumbers= [6, 7, 14, 17, 18, 42, 43, 44, 45, 46, 47, 48]
categorizingAttributeNumber= 28
numRows = 8000 # ChAnGe to 10000
# DATASET 2 but comparable to DATASET 1 in numRows
def dataset2PT2():
	global fileIn
	global fileOut
	global attributeNumbers
	global categorizingAttributeNumber
	global numRows
	fileIn= "AirQData/epa_air_quality_annual_summary.csv"
	fileOut= "AirQ_COMPARABLE_"
	attributeNumbers= [6, 7, 14, 17, 18, 42, 43, 44, 45, 46, 47, 48]
	categorizingAttributeNumber= 28
	numRows = 1000 # ChAnGe to 10000
'''
