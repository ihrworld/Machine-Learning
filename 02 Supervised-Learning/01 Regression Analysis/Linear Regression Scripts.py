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

# Build a linear regression model, using the Scikit-learn library
reg = LinearRegression() # Create the Linear Regression estimator
reg.fit ([[0, 1], [1, 1], [2, 1]], [0, 1, 2])  # Perform the fitting
reg.coef_  # Store the status of the estimator

#Split the data into training set and testing set
# "Training set" to train the system
# "Test set" to evaluate the learned or trained system
train_size = int(X_boston.shape[0]/2)
X_train = X_boston[:train_size]
X_test = X_boston[train_size:]
y_train = y_boston[:train_size]
y_test = y_boston[train_size:]
print('Train and test sizes {} {}'.format(X_train.shape, X_test.shape))

#Calculate the coefficients and the intercept
regr_boston = LinearRegression()  # Create the Linear Regression estimator
regr_boston.fit(X_train, y_train) # Perform the fitting
print('"Coeff" and "intercept": {} {}'.format(regr_boston.coef_, regr_boston.intercept_))

#Visualization of the coefficients and the intercerpt
from pandas import Series
# Plotting model coefficients:
coef = Series(regr_boston.coef_, boston.feature_names).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

#Compute the Score (𝑅2) and MSE (Mean Squared Error) for training and testing sets
print('Training Score: {}'.format(regr_boston.score(X_train, y_train)))
print('Testing Score: {}'.format(regr_boston.score(X_test, y_test)))
print('Training MSE: {}'.format(np.mean((regr_boston.predict(X_train) - y_train)**2)))
print('Testing MSE: {}'.format(np.mean((regr_boston.predict(X_test) - y_test)**2)))

#Regularization with Ridge Regression
regr_ridge = linear_model.Ridge(alpha=.3) # Create a Ridge regressor
regr_ridge.fit(X_train, y_train)  # Perform the fitting
print('Coeff and intercept: {} {}'.format(regr_ridge.coef_,  regr_ridge.intercept_))

#Regularization with Lasso Regression
regr_lasso = linear_model.Lasso(alpha=.3) # Create a Lasso regressor
regr_lasso.fit(X_train, y_train)  # Perform the fitting
print('Coeff and intercept: {} {}'.format(regr_lasso.coef_,  regr_lasso.intercept_))

#After appling Ridge and Lasso regression, the values of Score and MSE should be smaller

#Review the shape of the data and the features name:
print(boston.data.shape)
print(boston.target.shape)
print(boston.feature_names)

#Print the Max/Min/Mean price:
print('Max price {}, min price {}, and mean price {}'.format(np.max(boston.target), np.min(boston.target), np.mean(boston.target)))

#Identify what are the non important variables for the analysis.
ind=np.argsort(np.abs(regr_lasso.coef_))
print('Order variable (from less to more important): {}'.format(boston.feature_names[ind]))

indexes_non_selected=[0,2,3,4]
print('Non important variable: {}'.format(boston.feature_names[indexes_non_selected]))
most_important_index=[5]
print('Most important variable: {}'.format(boston.feature_names[most_important_index]))

#Select the most important features (K best features) using the module feature_selection of sklearn
import sklearn.feature_selection as fs 
# Create and fit selector
selector = fs.SelectKBest(score_func=fs.f_regression,k=5) # Select features according to the k highest scores.
X_new = selector.fit(X_train,y_train) # Perform the fitting
# Show the selected features (False => Non important variables; True => Important variables)
list(zip(selector.get_support(), boston.feature_names))


#Evaluate the predictions by visualizing prediction errors:
import matplotlib.pylab as plt
%matplotlib inline

regr_boston_all = LinearRegression() # Create the Linear Regression estimator
regr_boston_all.fit(boston.data, boston.target) # Fitting with all the data (not just the training data) and all the features
predicted = regr_boston_all.predict(boston.data) # Perform prediction of all the data

# Visualization of target and predicted responses of the boston data:
plt.scatter(boston.target, predicted, alpha=0.3)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')

x_plot = plt.scatter(predicted, (predicted - boston.target), c='b')
plt.hlines(y=0, xmin= 0, xmax=50)
plt.title('Residual plot')

#Model evaluation:
print('Score: {}'.format(regr_boston_all.score(boston.data, boston.target)))  # Best possible score is 1.0, lower values are worse.

#Compute Score using a single feature:
regr_feat1 = LinearRegression()
for i in np.arange(13):
    feat1=X_train[:,i:i+1]
    regr_feat1.fit(feat1, y_train)    
    print('Feature: {}'.format(boston.feature_names[i]))
    print('Score: {}'.format(regr_feat1.score(feat1, y_train)))
