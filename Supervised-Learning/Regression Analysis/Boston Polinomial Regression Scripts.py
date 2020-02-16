#Polinomial regression analysis, based on linear regression

# Import Libraries
import numpy as np  # multi-dimensional arrays manipulating
import pandas as pd  # data management and data analysis
import sklearn # machine learning library
import matplotlib.pylab as pl  # data visualization
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# Explore Sklearn datasets => Boston Housing Price
from sklearn import datasets
boston = datasets.load_boston() # Dictionary-like object that exposes its keys as attributes.
X_boston,y_boston = boston.data, boston.target # Create "X" matrix and "y" vector from the dataset.

# X_boston = store the data
# y_boston = store the classes/targets
print('Shape of data: {} {}'.format(X_boston.shape, y_boston.shape)) #Explore the shape of data (total records, total columns)

# Check the content of the dataset => the "keys or features" of the attributes and the general description
print('keys: {}'.format(boston.keys()))
print('feature names: {}'.format(boston.feature_names))
print(boston.DESCR)


#split the dataset into training set and test set
train_size = int(X_boston.shape[0]/2)
X_train = X_boston[:train_size]
X_test = X_boston[train_size:]
y_train = y_boston[:train_size]
y_test = y_boston[train_size:]
print('Train and test sizes {} {}'.format(X_train.shape, X_test.shape))


# Evaluation of the linear model
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

reg = LinearRegression() # Create the Linear Regression estimator
reg.fit ([[0, 1], [1, 1], [2, 1]], [0, 1, 2])  # Perform the fitting
reg.coef_  # Store the status of the estimator

regr_boston = LinearRegression()  # Create the Linear Regression estimator
regr_boston.fit(X_train, y_train) # Perform the fitting
print('"Coeff" and "intercept": {} {}'.format(regr_boston.coef_, regr_boston.intercept_))

#Compute Score ( 𝑅2 ) for training and testing sets
print('Training Score: {}'.format(regr_boston.score(X_train, y_train)))
print('Testing Score: {}'.format(regr_boston.score(X_test, y_test)))

#Compute MSE (Mean Squared Error) for training and testing sets
print('Training MSE: {}'.format(np.mean((regr_boston.predict(X_train) - y_train)**2)))
print('Testing MSE: {}'.format(np.mean((regr_boston.predict(X_test) - y_test)**2)))


# Evaluation of the polynomial model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create the Polynomial Regression estimator
regr_pol = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
#Perform the fit
regr_pol.fit(X_boston, y_boston) 

#print('Coeff and intercept: {} {}'.format(regr_pol.named_steps['linear'].coef_, regr_pol.named_steps['linear'].intercept_))
print('Multiple Polynomial regression Score: {}'.format(regr_pol.score(X_boston, y_boston)))
print('Multiple Polynomial regression MSE: {}'.format(np.mean((regr_pol.predict(X_boston) - y_boston)**2)))

#Quantitative evaluation of the SIMPLE lineal and polynomial regression
BostonDF = pd.DataFrame(boston.data)
BostonDF.head()
#Include the feature_names as columns
BostonDF.columns = boston.feature_names
BostonDF.head()

#Select the variable to analyze from the boston dataframe
x=BostonDF['LSTAT']
y=boston.target
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

#Make the prediction with linear regression
regr_boston = LinearRegression() # Create the Linear Regression estimator
regr_boston.fit(x, y) #Perform the fitting 

print('Simple linear regression Score: {}'.format(regr_boston.score(x, y)))
print('Simple linear regression MSE: {}'.format(np.mean((regr_boston.predict(x) - y)**2)))


#Make the prediction with polinomial regression
#Polinomial regression of order 2
regr_pol = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))]) 
regr_pol.fit(x, y) 

print('Simple Polynomial regression (order 2) Score: {}'.format(regr_pol.score(x, y)))
print('Simple Polynomial regression (order 2) MSE: {}'.format(np.mean((regr_pol.predict(x) - y)**2)))

#Polinomial regression of order 3
regr_pol = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
regr_pol.fit(x, y) 

print('Simple Polynomial regression (order 3) Score: {}'.format(regr_pol.score(x, y)))
print('Simple Polynomial regression (order 3) MSE: {}'.format(np.mean((regr_pol.predict(x) - y)**2)))
