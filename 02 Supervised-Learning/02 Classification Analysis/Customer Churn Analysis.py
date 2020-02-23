# Title: Churn Customer Analysis

# 1 Import Requiered Packages
import numpy as np #vectors and arrays operations
import pandas as pd  #data management and data analysis
import seaborn as sn # data visualization
import matplotlib.pyplot as plt # data visualization
import sklearn #machine learning library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report #sklearn metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import neighbors
from sklearn import metrics
#standarscaler => pre-processing
#open a new window and display the graphics in that way
%matplotlib inline 

# 2 Explore the data
#load the dataset
import pandas as pd 
churn = pd.read_csv('churn.csv')
churn.head()
#check all the title columns of the dataset
col_names = churn.columns.tolist()
print ("Column names:")
print (col_names)
#build the sample to analyse
to_show = col_names[:6] + col_names[-6:]
print ("\nSample data:")
churn[to_show].head(6)

# 3 Data Processing
# 3.1 Convert Categorical data to Boolean
#Convert Churn? data to boolean
#False. =>> 0
#True. =>> 1
churn = pd.get_dummies(churn, columns = ['Churn?'], drop_first = True)
churn.head(15)

#Convert 'Int'l Plan' and 'VMail Plan' data to boolean
#no =>> 0
#yes =>> 1
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn = pd.get_dummies(churn, columns = yes_no_cols, drop_first = True)
churn.head(15)

#remove unnecessary columns
to_drop = ['State','Phone']
churn_df = churn.drop(to_drop, axis = 1)
churn_df.head()

# 3.2 Response and Feature Variables 
X = churn_df.drop('Churn?_True.', axis = 1) # X variable = store the data
y = churn_df['Churn?_True.'] # y variable = store the classes/targets
X.head()
y.head()
print ("Feature space holds %d observations and %d features" % X.shape)
print ("Unique target labels:", np.unique(y))

#Visualization of the churn rate
%matplotlib inline
import matplotlib.pyplot as plt
plt.pie(np.c_[len(y)-np.sum(y),np.sum(y)][0],labels=['No Churn','Churn'],colors=['r','g'],shadow=True,autopct ='%.2f' )
fig = plt.gcf()
fig.set_size_inches(6,6)

# 3.3 Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape
X_test.shape

#Applying Standard Scaling to get optimized results
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #perform the transformation
X_test = sc.transform(X_test)

# 4 Prediction and Evaluation
# 4.1 Nearest Neighbours
#choose the best value for K
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

#build the model
n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
#predict the values for the test data 
y_pred = knn.predict(X_test)
#viusalize the confusion matrix and classification report
def draw_confusion(y_test,y_pred,labels):
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(cm)
    plt.title('Confusion matrix',size=20)
    ax.set_xticklabels([''] + labels, size=20)
    ax.set_yticklabels([''] + labels, size=20)
    plt.ylabel('Predicted',size=20)
    plt.xlabel('True',size=20)
    for i in range(2):
        for j in range(2):
            ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
    fig.set_size_inches(7,7)
    plt.show()
draw_confusion(y_test,y_pred,['no churn', 'churn'])
print (metrics.classification_report(y_test,y_pred))


# 4.2 Decision Tree
# Desired number of "folds" that we will do
cv = KFold(n_splits=5) 
accuracies = list()
max_attributes = len(list(churn_df))
depth_range = range(1, max_attributes + 1)
# Test the depth
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = depth,
                                             class_weight={1:3.5})
    for train_fold, valid_fold in cv.split(churn_df):
        f_train = churn_df.loc[train_fold] 
        f_valid = churn_df.loc[valid_fold] 
 
        model = tree_model.fit(X = f_train.drop(['Churn?_True.'], axis=1), 
                               y = f_train["Churn?_True."]) 
        valid_acc = model.score(X = f_valid.drop(['Churn?_True.'], axis=1), 
                                y = f_valid["Churn?_True."]) # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)
 
    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
# Show the results
df_results = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df_results = df_results[["Max Depth", "Average Accuracy"]]
print(df_results.to_string(index=False))
 
# Build the decession tree with a depth of 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth = 6,
                                            class_weight={1:3.5})
#Apply the Decession Tree Classifier
decision_tree.fit(X_train, y_train)
#predict the values for the test data 
y_pred = decision_tree.predict(X_test)

#viusalize the confusion matrix and classification report
import matplotlib.pyplot as plt
def draw_confusion(y_test,y_pred,labels):
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(cm)
    plt.title('Confusion matrix',size=20)
    ax.set_xticklabels([''] + labels, size=20)
    ax.set_yticklabels([''] + labels, size=20)
    plt.ylabel('Predicted',size=20)
    plt.xlabel('True',size=20)
    for i in range(2):
        for j in range(2):
            ax.text(i, j, cm[i,j], va='center', ha='center',color='white',size=20)
    fig.set_size_inches(7,7)
    plt.show()
draw_confusion(y_test,y_pred,['no churn', 'churn'])
print (metrics.classification_report(y_test,y_pred))

# Export the model to a .dot
import os  #manipulate data and directories structures (read and write files)
from sklearn.tree import export_graphviz #method that allows export the decession tree results to a dot file
from pydotplus import graph_from_dot_data #build the graph and export to PNG
dot_data = export_graphviz(decision_tree,
                              out_file= 'churn.dot',
                              max_depth = 6,
                              impurity = True,
                              feature_names = list(churn_df.drop(['Churn?_True.'], axis=1)),
                              rounded = True,
                              filled= True )
#convert dot to png
import os
os.system("dot -Tpng churn.dot -o churn.png")
#visualize the decesion tree
from IPython.core.display import Image
Image("churn.png")