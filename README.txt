Welcome to Joshua Santillo's OoM Machine Learning Classifier! 

For all: please `sudo pip3 install scikit-learn numpy matplotlib scipy itertools` to make sure you have all the required libraries!

IF	You're my lovely Graduate TA, please do the following:
	Open `config.py`
	Uncomment the first block comment around DATASET 1 and save
	Run `python3 analyze.py` in your terminal/command prompt
	Once you see the endText, "End of Joshua Santillo's HW 1", there should be a set of .csv files in Output
	Open `config.py`
	Uncomment the second block comment around DATASET 2 and save
	Run `python3 analyze.py` in your terminal/command prompt
	Once you see the endText, "End of Joshua Santillo's HW 1", there should be another set of .csv files in Output
	And you're done! You've got results # TODO, auto-generate graphs
FOR ANYONE ELSE: Welcome to Joshua Santillo's magic Order-of-Magnitude Classifier! This tools allows you to take .csv datasets of numeric attributes and use it to classify information by orders of magnitude. IT SIMPLY CAN'T DO ANYTHING ELSE :) You are welcome to modify this to suit your machine learning needs. It will run:
    Decision trees with some form of pruning
    Neural networks
    Boosting
    Support Vector Machines # TODO TODO
    k-nearest neighbors
Just how Dr. Isbell prescribes
I still don't know how to effectively apply the whole learning curve thing or SVM so just leave that commented (for your own good), a fix will be released in a later `commit`. For any other problems, please create an issue at https://github.com/sohuja96/Magic-OoM-Classifier/issues