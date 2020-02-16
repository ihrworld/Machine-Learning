# Title: Iris Flower Challenge - SVM Classification Algorithm

# 1 Import Necessary Libraries
import pandas as pd
import pandas as pd  #data management and data analysis
import seaborn as sn # data visualization
import matplotlib.pyplot as plt # data visualization
import sklearn #machine learning library
from sklearn import svm # support vector machine
from sklearn.svm import SVC #support vector classifier
from sklearn.metrics import confusion_matrix, classification_report #sklearn metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
#standarscaler => pre-processing
#open a new window and display the graphics in that way
%matplotlib inline 

# 2 Explore the data
#load the dataset
from sklearn.datasets import load_iris
iris = load_iris()
#Check the "keys or features" of the attributes and the general description:
print('keys: {}'.format(iris.keys())) 
print('feature names: {}'.format(iris.feature_names))
print(iris.DESCR)
#convert the data in a data frame
dfIris = pd.DataFrame(iris.data, columns = iris.feature_names)
dfIris.head()
##define the target
dfIris['target'] = iris.target
dfIris.head()
#check the names of the targets
iris.target_names
#create the lambda funtion
dfIris['flower_name'] = dfIris.target.apply(lambda x: iris.target_names[x])
dfIris.head()

# 3 Data Visualization
#data frame for setosa target
dfIris0 = dfIris[dfIris.target==0]
dfIris0.head()
#data frame for setosa versicolor
dfIris1 = dfIris[dfIris.target==1]
dfIris1.head()
#data frame for setosa virginica
dfIris2 = dfIris[dfIris.target==2]
dfIris2.head()
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(dfIris0['sepal length (cm)'], dfIris0['sepal width (cm)'])
plt.scatter(dfIris1['sepal length (cm)'], dfIris1['sepal width (cm)'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(dfIris0['petal length (cm)'], dfIris0['petal width (cm)'])
plt.scatter(dfIris1['petal length (cm)'], dfIris1['petal width (cm)'])

# 4 Prediction and Evaluation
# 4.1 Response and Feature Variables
X = dfIris.drop(['target', 'flower_name'], axis = 'columns')   # X variable = store the data
y = dfIris.target    # y variable = store the classes/targets
X.head()

# 4.2 Training and Test Split
#Use the 20% of the data for test and the 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

# 4.3 Apply Support Vector Machine Classifier
# 4.3.1 Simple Support Vector Machine Classifier
#Set up the SVM Classifier algorithm 
model =svm.SVC()
#Build the model on training data
model.fit(X_train, y_train)
#Predict the values of test group
pred_model = model.predict(X_test)
#Compute the score of the model
model.score(X_test, y_test)
#classification report
print(classification_report(y_test, pred_model))
#confussion matrix
print(confusion_matrix(y_test, pred_model))

# 4.3.2 Support Vector Machine Classifier with Kernel
#Set up the SVM Classifier algorithm 
model1 =SVC(kernel = 'linear')
#Build the model on training data
model1.fit(X_train, y_train)
#Predict the values of test group
pred_model1 = model1.predict(X_test)
#Compute the score of the model
model1.score(X_test, y_test)
#classification report
print(classification_report(y_test, pred_model1))
#confussion matrix
print(confusion_matrix(y_test, pred_model1))

# 4.4 Prediction with new values
#Set up the new values to analyze
X_new = [[7.0, 4.3, 2.2, 1.5,]]
#Predict the values of the test group applying the model with better accuracy rate
y_new = model1.predict(X_new)
y_new