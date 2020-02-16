# Boston Housing Price Prediction
# Make predictions with Visualization

# Histogram of prices
plt.hist(y_boston) 
plt.xlabel('price ($1000s)')
plt.ylabel('count')

# Visualization of the relations with scatter plots

#Linear Regression
# Example 1 => Visualization of the relations between price and LSTAT
df_boston = pd.DataFrame(boston.data, columns=boston.feature_names) # Create a DataFrame from the Boston dataset
df_boston['price'] = boston.target
sns.lmplot("price", "LSTAT", df_boston, aspect=2);
# Example 2 => Visualization of the relations between price and "average number of rooms per dwelling"
sns.lmplot("price", "RM", df_boston, aspect=2);
# Example 3 => Visualization of the relations between price and "average number of rooms per dwelling"
sns.lmplot("price", "AGE", df_boston, aspect=2);

#Polinomial Regression
# Example 1 => Estimate a polynomial regression of order 3
sns.lmplot("price", "LSTAT", df_boston, order=3, aspect=2);

# Heatmaps
# Plot just 7 of the 13 variables: PRICE, LSTAT, AGE, RM, NOX, INDUS, CRIM.
indexes=[0,2,4,5,6,12]
df_boston2 = pd.DataFrame(boston.data[:,indexes], columns=boston.feature_names[indexes])
df_boston2['PRICE'] = boston.target
corrmat = df_boston2.corr()
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)

# Scatter Plots Matrix
# Define the indexes
print(boston.feature_names)
indexes3=[5,6,12]
print(boston.feature_names[indexes3])
#build the scatter plots matrix
df_boston3 = pd.DataFrame(boston.data[:,indexes3], columns=boston.feature_names[indexes3])
df_boston3['price'] = boston.target
sns.pairplot(df_boston3) # Plot pairwise relationships in the dataset
