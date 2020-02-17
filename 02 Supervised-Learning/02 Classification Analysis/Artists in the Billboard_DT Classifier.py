# Title: 

# 1 Import Required Packages/Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.core.display import Image
from sklearn.tree import export_graphviz

# 2 Explore the data 
#Load the data set
df = pd.read_csv('artists_billboard.csv', sep=',')
df.head() 
df.shape

# 3 Data Visualization
#visualization of the TOP data
sb.factorplot('top',data=df,kind="count", aspect=1)
#visualization of the Artist_type data
sb.factorplot('artist_type',data=df,kind="count", aspect=1)
#visualization of the Mood data
sb.factorplot('mood',data=df,kind="count", aspect = 4)
#visualization of the Time data
sb.factorplot('tempo', data=df, hue='top', kind="count")
#visualization of the Genre
sb.factorplot('genre',data=df,kind="count", aspect=3)
#Visualization of the artist's "Year of birth"
sb.factorplot('anioNacimiento',data=df,kind="count", aspect=3)

# 4 Data Processing
# 4.1 Age_in_billboard Calculation
#replace 0 values for None
def age_fix(year):
    if year==0:
        return None
    return year
df['anioNacimiento']=df.apply(lambda x: age_fix(x['anioNacimiento']), axis=1);

#calculate the "age_in_billboard" feature values
def age_calculation(year,when):
    cad = str(when)
    moment = cad[:4]
    if year==0.0:
        return None
    return int(when) - year
df['age_in_billboard']=df.apply(lambda x: age_calculation(x['anioNacimiento'],x['chart_date']), axis=1);

#calculate the mean and std of the values under "age_in_billboard" column
age_avg = df['age_in_billboard'].mean()
age_std = df['age_in_billboard'].std()
age_null_count = df['age_in_billboard'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
conNullValues = np.isnan(df['age_in_billboard'])
df.loc[np.isnan(df['age_in_billboard']), 'age_in_billboard'] = age_null_random_list
df['age_in_billboard'] = df['age_in_billboard'].astype(int)
print("Average Age: " + str(age_avg))
print("Standard Deviation Age: " + str(age_std))
print("Interval to assign random agea: " + str(int(age_avg - age_std)) + " a " + str(int(age_avg + age_std)))

# 4.2 Mapping Data
# Mood Mapping 
df['moodEncoded'] = df['mood'].map( {'Energizing': 6, 'Empowering': 6, 'Cool': 5, 'Yearning': 4, 'Excited': 5, 'Defiant': 3, 'Sensual': 2, 'Gritty': 3, 
	                                 'Sophisticated': 4, 'Aggressive': 4, 'Fiery': 4, 'Urgent': 3, 'Rowdy': 4, 'Sentimental': 4, 'Easygoing': 1, 
	                                 'Melancholy': 4, 'Romantic': 2, 'Peaceful': 1, 'Brooding': 4, 'Upbeat': 5, 'Stirring': 5, 'Lively': 5, 'Other': 0,'':0} ).astype(int)
# Time Mapping 
df['tempoEncoded'] = df['tempo'].map( {'Fast Tempo': 0, 'Medium Tempo': 2, 'Slow Tempo': 1, '':0} ).astype(int)
# Genre Mapping 
df['genreEncoded'] = df['genre'].map( {'Urban': 4, 'Pop': 3, 'Traditional': 2, 'Alternative & Punk': 1,'Electronica': 1, 'Rock': 1, 'Soundtrack': 0, 
                                          'Jazz': 0, 'Other':0,'':0}).astype(int)                                 
# artist_type Mapping 
df['artist_typeEncoded'] = df['artist_type'].map( {'Female': 2, 'Male': 3, 'Mixed': 1, '': 0} ).astype(int)
# Mapping age when the artists reach the billboard
df.loc[ df['age_in_billboard'] <= 21, 'edadEncoded']                         = 0
df.loc[(df['age_in_billboard'] > 21) & (df['age_in_billboard'] <= 26), 'edadEncoded'] = 1
df.loc[(df['age_in_billboard'] > 26) & (df['age_in_billboard'] <= 30), 'edadEncoded'] = 2
df.loc[(df['age_in_billboard'] > 30) & (df['age_in_billboard'] <= 40), 'edadEncoded'] = 3
df.loc[ df['age_in_billboard'] > 40, 'edadEncoded'] = 4
# Mapping Song Duration
df.loc[ df['durationSeg'] <= 150, 'durationEncoded']                          = 0
df.loc[(df['durationSeg'] > 150) & (df['durationSeg'] <= 180), 'durationEncoded'] = 1
df.loc[(df['durationSeg'] > 180) & (df['durationSeg'] <= 210), 'durationEncoded'] = 2
df.loc[(df['durationSeg'] > 210) & (df['durationSeg'] <= 240), 'durationEncoded'] = 3
df.loc[(df['durationSeg'] > 240) & (df['durationSeg'] <= 270), 'durationEncoded'] = 4
df.loc[(df['durationSeg'] > 270) & (df['durationSeg'] <= 300), 'durationEncoded'] = 5
df.loc[ df['durationSeg'] > 300, 'durationEncoded'] = 6
#remove the elements that we do not need
drop_elements = ['id','title','artist','mood','tempo','genre','artist_type','chart_date','anioNacimiento','durationSeg','age_in_billboard']
artists_encoded = df.drop(drop_elements, axis = 1)
artists_encoded.head()
artists_encoded.shape

#4.3 Data Mapping Display
#In the column sum we will find the top ones, since being value 0 or 1, only those that did reach number 1 will be added.
#Check how the top=1 are splitted along the moodEncoded feature
artists_encoded[['moodEncoded', 'top']].groupby(['moodEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#Check how the top=1 are splitted along the artist_typeEncoded feature
artists_encoded[['artist_typeEncoded', 'top']].groupby(['artist_typeEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#Check how the top=1 are splitted along the genreEncoded feature
artists_encoded[['genreEncoded', 'top']].groupby(['genreEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#Check how the top=1 are splitted along the tempoEncoded feature
artists_encoded[['tempoEncoded', 'top']].groupby(['tempoEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#Check how the top=1 are splitted along the durationEncoded feature
artists_encoded[['durationEncoded', 'top']].groupby(['durationEncoded'], as_index=False).agg(['mean', 'count', 'sum'])
#Check how the top=1 are splitted along the edadEncoded feature
artists_encoded[['edadEncoded', 'top']].groupby(['edadEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# 5 Prediction and Evaluation
# 5.1 Training and Test Split
x_train = artists_encoded.drop(['top'], axis=1).values    # X variable = store the data
y_train = artists_encoded['top']  # y variable = store the classes/targets

# 5.2 Apply the Decission Tree Classifier 
# Desired number of "folds" that we will do
cv = KFold(n_splits=10) 
accuracies = list()
max_attributes = len(list(artists_encoded))
depth_range = range(1, max_attributes + 1)
# Test the depth
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = depth,
                                             class_weight={1:3.5})
    for train_fold, valid_fold in cv.split(artists_encoded):
        f_train = artists_encoded.loc[train_fold] 
        f_valid = artists_encoded.loc[valid_fold] 
 
        model = tree_model.fit(X = f_train.drop(['top'], axis=1), 
                               y = f_train["top"]) 
        valid_acc = model.score(X = f_valid.drop(['top'], axis=1), 
                                y = f_valid["top"]) # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)
 
    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)  
# Show the results
df_results = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df_results = df_results[["Max Depth", "Average Accuracy"]]
print(df_results.to_string(index=False))

# 5.3 Decession Tree Visualization
# Build the decession tree with a depth of 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth = 4,
                                            class_weight={1:3.5})
#Apply the Decession Tree Classifier
decision_tree.fit(x_train, y_train)

# Export the model to a .dot
from sklearn.tree import export_graphviz #method that allows export the decession tree results to a dot file
from pydotplus import graph_from_dot_data #build the graph and export to PNG

dot_data = export_graphviz(decision_tree,
                              out_file= 'tree1.dot',
                              max_depth = 7,
                              impurity = True,
                              feature_names = list(artists_encoded.drop(['top'], axis=1)),
                              class_names = ['No', 'N1 Billboard'],
                              rounded = True,
                              filled= True )

import os #manipulate data and directories structures (read and write files)
os.system("dot -Tpng tree1.dot -o tree1.png")
#Visualize the decession tree
from IPython.core.display import Image
Image("arbol-decision-billboard.png")
#check the precission of the decession tree
accuracy = round(decision_tree.score(x_train, y_train) * 100, 2)
print(accuracy)

# 5.4 Predicting new values
#predict artist CAMILA CABELLO featuring YOUNG THUG
x_test = pd.DataFrame(columns=('top','moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
x_test.loc[0] = (1,5,2,4,1,0,3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis = 1))
print("Probabilidad de Acierto: " + str(round(y_proba[0][y_pred]* 100, 2))+"%")

#predict artist Imagine Dragons 
x_test = pd.DataFrame(columns=('top','moodEncoded', 'tempoEncoded', 'genreEncoded','artist_typeEncoded','edadEncoded','durationEncoded'))
x_test.loc[0] = (0,4,2,1,3,2,3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis = 1))
print("Probabilidad de Acierto: " + str(round(y_proba[0][y_pred]* 100, 2))+"%")