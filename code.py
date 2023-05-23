import pandas as pd
import numpy as np
import zipfile
import os
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier # To use as a baseline

from sklearn.feature_extraction.text import CountVectorizer # To process the comments, i.e. process the "text" objects to classify
from sklearn.model_selection import train_test_split # To split the dataset into train, validate, and test sets
from sklearn.decomposition import PCA
from sklearn import model_selection

# For pipeline hyperparameter tuning
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler # Needed for MultinomialNB


def loadData(vectorizor):
	"""
	Load the data and split it into training, test, and validation sets.

	Parameters: vectorizor - the CountVectorizer to transform the comments (which are the text inputs)

	Returns: The formed input and output sets as np arrays:
		x_train, x_test, x_test_holdout, x_val, y_train, y_test, y_test_holdout, y_val
	"""
	# Obtain the data from the zip file
	unzip = zipfile.ZipFile("YouTube-Spam-Collection-v1.zip")

	# Load each individual file's data
	dataPsy = pd.read_csv(unzip.open("Youtube01-Psy.csv"))
	dataKaty = pd.read_csv(unzip.open("Youtube02-KatyPerry.csv"))
	dataLMFAO = pd.read_csv(unzip.open("Youtube03-LMFAO.csv"))
	dataEminem = pd.read_csv(unzip.open("Youtube04-Eminem.csv"))
	dataShakira = pd.read_csv(unzip.open("Youtube05-Shakira.csv"))

	# Concatenate all the data, and only use the input and output values (i.e. the comment and its label)
	data = pd.concat([dataPsy, dataKaty, dataLMFAO, dataEminem, dataShakira])
	#print(data.sample(5)) # DEBUG
	x_total = np.array(data['CONTENT']) # The input: the comments to classify
	y_total = np.array(data['CLASS']) # where 1=spam, 0=not spam. Note: in dataset description CLASS = TAG

	# Split the data in 2 ways:
	#	For the baseline hold-out validation: split into 70% train, 15% test, and 15% validation set sizes.
	#	For k-fold validation used on the ML algorithms: split into 70% train/validation and 30% test set sizes.
	x_total = vectorizor.fit_transform(x_total)
	x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.3)
	x_test_holdout, x_val, y_test_holdout, y_val = train_test_split(x_test, y_test, test_size=0.5)

	# Ensure properly fit
	x_train = x_train.todense()
	x_test = x_test.todense()
	x_test_holdout = x_test_holdout.todense()
	x_val = x_val.todense()

	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)

	# Plot the original data
	pca = PCA(n_components=2)
	proj = pca.fit_transform(np.asarray(x_total.todense()))
	plt.scatter(proj[:, 0], proj[:, 1], c=np.asarray(y_total))
	plt.colorbar() 
	plt.tight_layout()
	plt.savefig('data_spread.jpg')

	return x_train, x_test, x_test_holdout, x_val, y_train, y_test, y_test_holdout, y_val


def modelPrediction(model, x_train, y_train, x_test, y_test, baselineAlgorithm=True):
	"""
	Fit and score the model and print the statistics.

	Parameters:
		model: the model to fit and score
		x_train: np array of the training set input
		y_train: np array of the training set output
		x_test: np array of the test set input
		y_test: np array of the test set output
		baselineAlgorithm: boolean indicating if the model is the baseline model

	Returns: modelScore - the evaluation metric of this trained model
			 crossValScore - the metric to visualize comparisons between models
	"""
	model.fit(x_train, y_train)
	predicted = model.predict(x_train)
	print(metrics.classification_report(y_train, predicted))
	print(metrics.confusion_matrix(y_train, predicted))
	modelScore = model.score(x_test,y_test)
	kfold = model_selection.KFold(n_splits=10)
	crossValScore = model_selection.cross_val_score(model, x_test, y_test, cv=kfold, scoring='accuracy')
	print("Cross Validation score: {}\n".format(crossValScore))

	if baselineAlgorithm:
		print("Model score: {}\n\n\n".format(modelScore)) # Note: holdout test sets
	else:
		print("Tuned parameters: {}\nModel score: {}\n\n\n".format(model.best_params_, modelScore))

	return modelScore, crossValScore


def baselineModel(x_train, x_test_holdout, x_val, y_train, y_test_holdout, y_val):
	"""
	Train and evaluate the baseline model, and print the statistics.
	The baseline model is a dummy classifier tuned according to the possible strategies:
		"stratified", "most_frequent", "prior", "uniform", or "constant"

	Parameters:
		x_train: np array of the training set input
		x_test_holdout: np array of the test set input
		x_val: np array of the validation set input
		y_train: np array of the training set output
		y_test_holdout: np array of the test set output
		y_val: np array of the validation set output

	Returns:
		model - the trained model
		modelScore - the evaluation metric of this trained model
		crossValScore - the metric to visualize comparisons between models
	"""
	baselineStrategies = ["stratified", "most_frequent", "prior", "uniform", "constant"]
	bestModel = "stratified"
	bestScore = 0
	for i in baselineStrategies:
		# Train the model
		if i == "constant":
			model = DummyClassifier(strategy=i, constant=0)
		elif (i == "stratified") or (i == "uniform"):
			model = DummyClassifier(strategy=i, random_state=50)
		else:
			model = DummyClassifier(strategy=i)
		model.fit(x_train, y_train)
		predicted = model.predict(x_train)

		# Run the validation test
		valScore = model.score(x_val,y_val)
		if valScore > bestScore:
			bestScore = valScore
			bestModel = i

	# Use the tuned model and compute the performance on the test set
	if bestModel == "constant":
		model = DummyClassifier(strategy=bestModel, constant=0)
	elif (bestModel == "stratified") or (bestModel == "uniform"):
		model = DummyClassifier(strategy=bestModel, random_state=50)
	else:
		model = DummyClassifier(strategy=bestModel)

	print(model)
	modelScore, crossValScore = modelPrediction(model, x_train, y_train, x_test_holdout, y_test_holdout, True)
	return model, modelScore, crossValScore


def multinomialNBModel(x_train, x_test, y_train, y_test):
	"""
	Train and evaluate the MultinomialNB model, and print the statistics.

	Parameters:
		x_train: np array of the training set input
		x_test: np array of the test set input
		y_train: np array of the training set output
		y_test: np array of the test set output

	Returns:
		model - the trained model
		modelScore - the evaluation metric of this trained model
		crossValScore - the metric to visualize comparisons between models
	"""
	# Use a pipeline to tune the hyperparameters (alpha) using k-fold validation
	bb_pipe = make_pipeline( SimpleImputer(strategy="median"), MinMaxScaler(), MultinomialNB() )
	param_grid = { "multinomialnb__alpha": [0.001, 0.1, 1, 10, 100] }
	model = GridSearchCV(bb_pipe, param_grid, scoring='accuracy', cv=3, return_train_score=True, verbose=0, n_jobs=-1)
	print(MultinomialNB())
	modelScore, crossValScore = modelPrediction(model, x_train, y_train, x_test, y_test, False)

	return model, modelScore, crossValScore


def kNNModel(x_train, x_test, y_train, y_test):
	"""
	Train and evaluate the k-Nearest Neighbor model, and print the statistics.

	Parameters:
		x_train: np array of the training set input
		x_test: np array of the test set input
		y_train: np array of the training set output
		y_test: np array of the test set output

	Returns:
		model - the trained model
		modelScore - the evaluation metric of this trained model
		crossValScore - the metric to visualize comparisons between models
	"""
	# Use a pipeline to tune the hyperparameters (n_neighbors and weights) using k-fold validation
	bb_pipe = make_pipeline( SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier() )
	param_grid = { "kneighborsclassifier__n_neighbors": [1, 5, 10, 20, 30, 40, 50, 100],
	    "kneighborsclassifier__weights": ['uniform', 'distance'] }
	model = GridSearchCV(bb_pipe, param_grid, cv=3, verbose=0, n_jobs=-1)
	print(KNeighborsClassifier())
	modelScore, crossValScore = modelPrediction(model, x_train, y_train, x_test, y_test, False)

	return model, modelScore, crossValScore


def SVMModel(x_train, x_test, y_train, y_test):
	"""
	Train and evaluate the Support Vector Machine model, and print the statistics.

	Parameters:
		x_train: np array of the training set input
		x_test: np array of the test set input
		y_train: np array of the training set output
		y_test: np array of the test set output

	Returns:
		model - the trained model
		modelScore - the evaluation metric of this trained model
		crossValScore - the metric to visualize comparisons between models
	"""
	# Use a pipeline to tune the hyperparameters (gamma and c) using k-fold validation
	bb_pipe = make_pipeline( SimpleImputer(strategy="median"), StandardScaler(), SVC() )
	param_grid = { "svc__gamma": [0.01, 0.1, 1.0, 10], "svc__C": [0.1, 1.0, 10, 100] }
	model = GridSearchCV(bb_pipe, param_grid, cv=3, return_train_score=True, verbose=0, n_jobs=-1)
	print(SVC())
	modelScore, crossValScore = modelPrediction(model, x_train, y_train, x_test, y_test, False)

	return model, modelScore, crossValScore


def DTCModel(x_train, x_test, y_train, y_test):
	"""
	Train and evaluate the Decision Tree Classifier model, and print the statistics.

	Parameters:
		x_train: np array of the training set input
		x_test: np array of the test set input
		y_train: np array of the training set output
		y_test: np array of the test set output

	Returns:
		model - the trained model
		modelScore - the evaluation metric of this trained model
		crossValScore - the metric to visualize comparisons between models
	"""
	# Use a pipeline to tune the hyperparameters (criterion, max_depth, and min_samples_leaf) using k-fold validation
	bb_pipe = DecisionTreeClassifier(random_state=123)
	param_grid = { "criterion": ['gini', 'entropy'], "max_depth": [2,4,6,8,10,12], "min_samples_leaf": [1, 2, 3, 4, 5] }
	model = GridSearchCV(bb_pipe, param_grid, cv=3, verbose=0, n_jobs=-1)
	print(DecisionTreeClassifier())
	modelScore, crossValScore = modelPrediction(model, x_train, y_train, x_test, y_test, False)

	return model, modelScore, crossValScore


def useBestModel(bestModel, results, vectorizor):
	"""
	Plot comparisons of models, and use the best model to make new predictions.

	Parameters:
		bestModel: the model to use for new predictions
		results: list of the cross validation scores of all the models
		vectorizor: the initialized CountVectorizer
	"""
	
	# Compare the different model algorithms using box plots
	names = ["Baseline", "MNB", "kNN", "SVM", "DTC"]

	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.savefig('compare_models.jpg')

	# Input new comments to predict as either spam or not.
	# Note: surround the input by quotation marks, e.g.: "Example test comment!"
	while True:
		commentToPredict = input("Enter a comment (enclosed with quotation marks) to predict as either spam or not spam: ")

		if (len(commentToPredict) > 2) and (commentToPredict[0] == commentToPredict[-1] == '"'):
			formattedInput = vectorizor.transform([commentToPredict]).toarray()
			spam = bestModel.predict(formattedInput)

			if spam[0]:
				print("Predicted as: spam")
			else:
				print("Predicted as: not spam")

			# Ask the user to continue
			continueLoop = input("Make another prediction? Enter \"y\", otherwise the program will exit: ")
			if continueLoop != "\"y\"":
				break
		else:
			print("\nError: please ensure the comment is enclosed with quotation marks, e.g.: \"Example test comment!\"")


def main():
	"""
	Train and evaluate the various models on the Youtube Spam Collection dataset.
	"""
	vectorizor = CountVectorizer()

	# Split the dataset into training, test, and validation sets
	x_train, x_test, x_test_holdout, x_val, y_train, y_test, y_test_holdout, y_val = loadData(vectorizor)

	# Maintain a dictionary where the keys are a model's score and the value is the model itself,
	# and maintain a list of the cross validation scores for the models
	modelsRanked = {}
	results = []

	# 0: the baseline: a dummy classifier tuned according to the possible strategies
	model, score, cvScore = baselineModel(x_train, x_test_holdout, x_val, y_train, y_test_holdout, y_val)
	modelsRanked[score] = model
	results.append(cvScore)

	# 1: Multinomial Naive Bayes
	model, score, cvScore = multinomialNBModel(x_train, x_test, y_train, y_test)
	modelsRanked[score] = model
	results.append(cvScore)

	# 2: k-Nearest Neighbor
	model, score, cvScore = kNNModel(x_train, x_test, y_train, y_test)
	modelsRanked[score] = model
	results.append(cvScore)

	# 3: Support Vector Machine
	model, score, cvScore = SVMModel(x_train, x_test, y_train, y_test)
	modelsRanked[score] = model
	results.append(cvScore)

	# 4: Decision Tree Classifier
	model, score, cvScore = DTCModel(x_train, x_test, y_train, y_test)
	modelsRanked[score] = model
	results.append(cvScore)

	# Use the best-scored model
	bestModel = modelsRanked[max(modelsRanked.keys())]
	useBestModel(bestModel, results, vectorizor)


if __name__ == "__main__":
	main()
