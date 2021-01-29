# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 00:41:55 2021

@author: arjun
"""

######################## BUILDING THE MODEL ########################
# compare algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

############################################################################
###################### MACHINE LEARNING PIPELINE
###########################################################################
# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
# Load dataset
"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
"""
iris = load_iris()
#Convert it to dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target_names'])
# input variable
X = df.drop(['target_names'], axis = 1) 
X.head(2)

# output variable
y = df['target_names'] 
y.head(2)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


"""# Save SV Classifier model using Pickel"""

## Pickle
import pickle

# save model
pickle.dump(model, open('iris_data_classification.pickle', 'wb'))

# load model
iris_data_classification_model = pickle.load(open('iris_data_classification.pickle', 'rb'))

# predict the output
y_pred = iris_data_classification_model.predict(X_validation)

#Predict output for two inputs
iris_data_classification_model.predict(X_validation.head(2))
iris_data_classification_model.predict([[1,1,1,1],[5.3,3.5,1.1,1.2]])

# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(Y_validation, y_pred),'\n')

# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(Y_validation, y_pred))

"""End ==================================================    <br>
This Project was Created By Khangjrakpam Arjun
"""