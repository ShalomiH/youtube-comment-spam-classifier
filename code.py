import pandas as pd
import numpy as np
import zipfile
import os

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier # To use as a baseline

from sklearn.feature_extraction.text import CountVectorizer # To process the comments, i.e. process the "text" objects to classify
from sklearn.model_selection import train_test_split # To split the dataset into train, validate, and test sets
from sklearn.metrics import confusion_matrix, classification_report

# For pipeline hyperparameter tuning
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler # Needed for MultinomialNB

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
vectorizor = CountVectorizer()
x_total = vectorizor.fit_transform(x_total)
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.3)
x_test_holdout, x_val, y_test_holdout, y_val = train_test_split(x_test, y_test, test_size=0.5)

# Ensure properly fit
x_train = x_train.todense()
x_test = x_test.todense()
x_test_holdout = x_test_holdout.todense()
x_val = x_val.todense()


# 0: the baseline: a dummy classifier tuned according to the possible strategies
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

model.fit(x_train, y_train)
print(model)
predicted = model.predict(x_train)
print(metrics.classification_report(y_train, predicted))
print(metrics.confusion_matrix(y_train, predicted))
print("Model score: {}\n\n\n".format(model.score(x_test_holdout,y_test_holdout))) # Uses the test sets


# 1: Multinomial Naive Bayes
# Use a pipeline to tune the hyperparameters (alpha) using k-fold validation
bb_pipe = make_pipeline( SimpleImputer(strategy="median"), MinMaxScaler(), MultinomialNB() )
param_grid = { "multinomialnb__alpha": [0.001, 0.1, 1, 10, 100] }
model = GridSearchCV(bb_pipe, param_grid, scoring='accuracy', cv=3, return_train_score=True, verbose=0, n_jobs=-1)
model.fit(x_train, y_train)
print(MultinomialNB())
predicted = model.predict(x_train)
print(metrics.classification_report(y_train, predicted))
print(metrics.confusion_matrix(y_train, predicted))
print("Tuned parameters: {}\nModel score: {}\n\n\n".format(model.best_params_, model.score(x_test,y_test)))


# 2: k-Nearest Neighbor
# Use a pipeline to tune the hyperparameters (n_neighbors and weights) using k-fold validation
bb_pipe = make_pipeline( SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier() )
param_grid = { "kneighborsclassifier__n_neighbors": [1, 5, 10, 20, 30, 40, 50, 100],
    "kneighborsclassifier__weights": ['uniform', 'distance'] }
model = GridSearchCV(bb_pipe, param_grid, cv=3, verbose=0, n_jobs=-1)
model.fit(x_train, y_train)
print(KNeighborsClassifier())
predicted = model.predict(x_train)
print(metrics.classification_report(y_train, predicted))
print(metrics.confusion_matrix(y_train, predicted))
print("Tuned parameters: {}\nModel score: {}\n\n\n".format(model.best_params_, model.score(x_test,y_test)))


# 3: Support Vector Machine
# Use a pipeline to tune the hyperparameters (gamma and c) using k-fold validation
bb_pipe = make_pipeline( SimpleImputer(strategy="median"), StandardScaler(), SVC() )
param_grid = { "svc__gamma": [0.01, 0.1, 1.0, 10], "svc__C": [0.1, 1.0, 10, 100] }
model = GridSearchCV(bb_pipe, param_grid, cv=3, return_train_score=True, verbose=0, n_jobs=-1)
model.fit(x_train, y_train)
print(SVC())
predicted = model.predict(x_train)
print(metrics.classification_report(y_train, predicted))
print(metrics.confusion_matrix(y_train, predicted))
print("Tuned parameters: {}\nModel score: {}".format(model.best_params_, model.score(x_test,y_test)))
