#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Fit aGaussian Naive Bayes 
nb = GaussianNB()  #set up the class
nb.fit(X_train,y_train) #perform the fit
#predict the values with the test data
y_pred = nb.predict(X_test)
from sklearn import metrics
#Define a function to visualize the results
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred), interpolation='nearest',cmap='gray')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    fig = plt.gcf()
    fig.set_size_inches(9,9)    
#Define the Confussion Matrix
print ("classification accuracy:", metrics.accuracy_score(pred_nb, y_test))
plot_confusion_matrix(pred_nb, y_test)
#Define the Classification Report
print ("Classification Report:")
print (metrics.classification_report(pred_nb,np.array(y_test)))



# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
#Fit a Multinomial Naive Bayes
nb = MultinomialNB()  #set up the class
nb.fit(X_train,y_train) #perform the fit
#predict the values with the test data
y_pred = nb.predict(X_test)
from sklearn import metrics
#Define a function to visualize the results
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred), interpolation='nearest',cmap='gray')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    fig = plt.gcf()
    fig.set_size_inches(9,9)    
#Define the Confussion Matrix
print ("classification accuracy:", metrics.accuracy_score(pred_nb, y_test))
plot_confusion_matrix(pred_nb, y_test)
#Define the Classification Report
print ("Classification Report:")
print (metrics.classification_report(pred_nb,np.array(y_test)))


# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
class sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
#Fit a Bernoulli Naive Bayes
nb = BernoulliNB()  #set up the class
nb.fit(X_train,y_train) #perform the fit
#predict the values with the test data
y_pred = nb.predict(X_test)
from sklearn import metrics
#Define a function to visualize the results
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred), interpolation='nearest',cmap='gray')
    plt.colorbar()
    plt.ylabel('true value')
    plt.xlabel('predicted value')
    fig = plt.gcf()
    fig.set_size_inches(9,9)    
#Define the Confussion Matrix
print ("classification accuracy:", metrics.accuracy_score(pred_nb, y_test))
plot_confusion_matrix(pred_nb, y_test)
#Define the Classification Report
print ("Classification Report:")
print (metrics.classification_report(pred_nb,np.array(y_test)))