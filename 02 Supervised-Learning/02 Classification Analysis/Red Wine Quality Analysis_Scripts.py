# Title: Red Wine Quality Analysis

# 1 Import Requiered Packages
import pandas as pd  #data management and data analysis
import seaborn as sn # data visualization
import matplotlib.pyplot as plt # data visualization
import sklearn #machine learning library
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm # support vector machine
from sklearn.svm import SVC #support vector classifier
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report #sklearn metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#standarscaler => pre-processing
#open a new window and display the graphics in that way
%matplotlib inline 

# 2 Explore the data
#load the dataset
wine = pd.read_csv('winequality-red.csv', sep = ',')
wine.head(10)
wine.info()

#check null values in the dataset
wine.isnull().sum()

# 3 Data Processing
#split the data frame in two bins related to the quality of the wine ('bad quality', 'good quality') 
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels = group_names)
wine['quality'].unique()

#Use the Label Encoder class to convert the categorical text data ('bad' and 'good') into model-understandable numerical data
label_quality = LabelEncoder()

#Transform the text into binary numerical data with the pre-processing method
wine['quality'] = label_quality.fit_transform(wine['quality']) #perform the transformation

#display the data
wine.head(10)

#count the number of records of the quality groups
wine['quality'].value_counts()

# 4 Prediction and Evaluation
# 4.1 Response and Feature Variables
X = wine.drop('quality', axis = 1)   # X variable = store the data
y = wine['quality']  # y variable = store the classes/targets

# 4.2 Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Applying Standard Scaling to get optimized results
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #perform the transformation
X_test = sc.transform(X_test)

# 4.3 Random Forest Classifier
#Set up the Random Forest algorithm and the number of trees that we are going to build
rfc = RandomForestClassifier(n_estimators = 250)
#Build the model on training data
rfc.fit(X_train, y_train)
#Predict the values of test group
pred_rfc = rfc.predict(X_test)
#classification report
print(classification_report(y_test, pred_rfc))
#confussion matrix
print(confusion_matrix(y_test, pred_rfc))

# 4.4 SVM Classifier
#Set up the SVM Classifier algorithm
clf =svm.SVC()
#Build the model on training data
clf.fit(X_train, y_train)
#Predict the values of test group
pred_clf = clf.predict(X_test)
#classification report
print(classification_report(y_test, pred_clf))
#confussion matrix
print(confusion_matrix(y_test, pred_clf))

# 4.5 Neuronal Networks
from sklearn.neural_network import MLPClassifier 
#Set up the Neural Network Classifier algorithm
nlpc = MLPClassifier(hidden_layer_sizes = (11, 11, 11), max_iter = 500)
#Build the model on training data
nlpc.fit(X_train, y_train)
#Predict the values of test group
pred_nlpc = clf.predict(X_test)
#classification report
print(classification_report(y_test, pred_nlpc))
#confussion matrix
print(confusion_matrix(y_test, pred_nlpc))

#4.6 Accuracy score
from sklearn.metrics import accuracy_score
#Accuracy score of Random Forest Classifier
cm = accuracy_score(y_test, pred_rfc)
cm
#Accuracy score of SVM Classifier
cm1 = accuracy_score(y_test, pred_clf)
cm1
#Accuracy score of Neuronal Network Classifier
cm2 = accuracy_score(y_test, pred_nlpc)
cm2

# 4.7 Prediction with new values
#Set up the new values to analyze
X_new = [[7.3, 0.58, 0.00, 2.0, 0.065, 15.0, 21.0, 0.9946, 3.36, 0.47, 10.]]
#Applying Standard Scaling to get optimized results
X_new = sc.transform(X_new)
#Predict the values of the test group
y_new = rfc.predict(X_new)
y_new
