# Title Credit Card Fraud Detection

#1 Import Necessay Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

from pylab import rcParams
from collections import Counter

#set up graphic style in this case I am using the color scheme from xkcd.com
rcParams['figure.figsize'] = 14, 8.7 # Golden Mean
LABELS = ["Normal","Fraud"]
#col_list = ["cerulean","scarlet"]# https://xkcd.com/color/rgb/
#sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list))

%matplotlib inline

#2 Explore the data
#load the dataset
df = pd.read_csv("creditcard.csv")
df.head(10)

#check the shape of the dataset
df.shape

#check for unbalanced values
df.groupby('Class').size()

#visualization of the Class data
sb.factorplot('Class',data=df,kind="count", aspect=2)

# 3 Predict and Evaluation
#3.1 Response and Feature Variables
X = df.drop('Class', axis = 1)   # X variable = store the data
y = df['Class']    # y variable = store the classes/targets
X.head()
y.head()

# 3.2 Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X_train.shape
X_test.shape

# 3.3 Logistic Regression Classifier
#create a function to adjust the unbalanced values
def run_model_balanced(X_train, X_test, y_train, y_test):
    #Set up the Logistic Regression algorithm
    clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    #build the model on training data
    clf.fit(X_train, y_train) 
    return clf
model = run_model_balanced(X_train, X_test, y_train, y_test)

#Predict the values of test group
pred_y = model.predict(X_test)

#create a function to visualize the results
def show_results(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

#Visualize the results
show_results(y_test, pred_y)

# 3.4 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#Set up the Random Forest Classifier with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,verbose=2,
                               max_features = 'sqrt')
#build the model on training data
model.fit(X_train, y_train)

#Predict the values of test group
pred_y = model.predict(X_test)

#Visualize the results
show_results(y_test, pred_y)

from sklearn.metrics import precision_score
#Compute the precission of the model precisión del modelo
precision = precision_score(y_test, pred_y)
print(precision)

from sklearn.metrics import recall_score
#check sensitivity/recall of the model
sensibility = recall_score(y_test, pred_y)
print(sensibility)

from sklearn.metrics import f1_score
#Check the F1-score (a combination between precision and sensibility)
f1_score = f1_score(y_test, pred_y)
print(f1_score)

from sklearn.metrics import roc_auc_score
#Check the ROC-AUC curve of the model
roc_auc = roc_auc_score(y_test, pred_y)
print(roc_auc)

