# Title: New York Times - News Classification - Naive Bayes Classifier

# 1 Import the necessary libraries
import pandas as pd  #data management and data analysis
import seaborn as sn # data visualization
import matplotlib.pyplot as plt # data visualization
import sklearn #machine learning library
from sklearn import svm # support vector machine
from sklearn.svm import SVC #support vector classifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report #sklearn metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#open a new window and display the graphics in that way
%matplotlib inline 

# 2 Explore the data
#Load the data
dataNYT=pd.read_csv('Boydstun_NYT_FrontPage_Dataset_1996-2006_0.csv')
dataNYT.head()
#check the data
dataNYT.info()
#Identify null values
dataNYT.isnull().sum()

# 3 Data Processing
# 3.1 Response and Feature Variables
X = dataNYT['Title']     # X variable = store the data
y = dataNYT['Topic_2digit']      # y variable = store the classes/targets

# 3.2 Training and Test Split
split = pd.to_datetime(pd.Series(dataNYT['Date']))<pd.datetime(2004, 1, 1) #data to train
#train and test split for X variable (data to analyze)
X_train = X[split]
X_test = X[np.logical_not(split)]
#train and test split for y variable (target)
y_train = y[split]
y_test = y[np.logical_not(split)]
print ('Check the split sizes, train, test and total amount of data:')
print (X_train.shape, X_test.shape, X.shape)
print ('Display the labels:')
print (np.unique(y)) #remove duplicated values

# 3.3 Data Tokenization
# Use the count number of instances considering that a word has a minimum support of two documents
vectorizer = CountVectorizer(min_df=2, 
# stop words such as 'and', 'the', 'of' are removed                             
stop_words='english', 
strip_accents='unicode')
#Fit and convert data
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print ("Number of tokens: " + str(len(vectorizer.get_feature_names())) +"\n")
print ("Extract of tokens:")
print (vectorizer.get_feature_names()[1000:1100])

# 4 Prediction and Evaluation

%matplotlib inline
#Fit a Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()  #set up the class
nb.fit(X_train,y_train) #perform the fit

pred_nb = nb.predict(X_test)
from sklearn import metrics
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred), interpolation='nearest',cmap='gray')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    fig = plt.gcf()
    fig.set_size_inches(9,9)    
print ("classification accuracy:", metrics.accuracy_score(pred_nb, y_test))
plot_confusion_matrix(pred_nb, y_test)
print ("Classification Report:")
print (metrics.classification_report(pred_nb,np.array(y_test)))

#Save data for future use
import pickle
ofname = open('NYT_data.pkl', 'wb')
s = pickle.dump([X_train,y_train,X_test,y_test],ofname)
ofname.close()

#What are the top N most predictive features per class?
N = 5
voc = vectorizer.get_feature_names()
for i, label in enumerate(np.unique(y)):
    topN = np.argsort(nb.coef_[i])[-N:]
    print ('Code: '+ str(label) + ' Terms : '+ str([voc[i] for i in topN]))

#Add more features to the model to get a better prediction result
X = dataNYT['Title']+dataNYT['Summary']
X_train = X[split]
X_test = X[np.logical_not(split)]
y = dataNYT['Topic_2digit']
y_train = y[split]
y_test = y[np.logical_not(split)]

# Tokenize the data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=2, 
# stop words such as 'and', 'the', 'of' are removed                                  
stop_words='english', 
strip_accents='unicode')
#Fit and convert data
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print ("\n")
print ("Number of tokens: " + str(len(vectorizer.get_feature_names())) +"\n")
print ("Extract of tokes:")
print( vectorizer.get_feature_names()[1000:1100])

%matplotlib inline
#Fit a Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()  #set up the class
nb.fit(X_train,y_train) #perform the fit

pred_nb = nb.predict(X_test)
from sklearn import metrics
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred), interpolation='nearest')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    fig = plt.gcf()
    fig.set_size_inches(9,9)    
print ("classification accuracy:", metrics.accuracy_score(pred_nb, y_test))
plot_confusion_matrix(pred_nb, y_test)
print ("Classification Report:")
print (metrics.classification_report(pred_nb,np.array(y_test)))

#Save data for future use.
import pickle
ofname = open('NYT_context_data.pkl', 'wb')
s = pickle.dump([X_train,y_train,X_test,y_test],ofname)
ofname.close()

#What are the top N most predictive features per class?
N = 5
voc = vectorizer.get_feature_names()
for i, label in enumerate(np.unique(y)):
    topN = np.argsort(nb.coef_[i])[-N:]
    print ('Code: '+ str(label) + ' Terms : '+ str([voc[i] for i in topN]))
