# Run this command to download the data
# Importing Libraries

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression


from sklearn.metrics import  confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("./data/adult_data.csv")
print('Dataframe shape: ', df.shape)
print(df.isnull().sum())
df.head()

# Data Cleanup

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(how='any',axis=0)

print('Dataframe shape: ', df.shape)
print(df.isnull().sum())
df.head()

# Visualize
import seaborn as sns

    # Visualizing amount of each income status
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="income", data=df, palette=sns.xkcd_palette(["azure", "light red"]))
plt.xlabel('Income')
plt.ylabel('Count')
# plt.savefig('./plots/income_count.png')
plt.show()

    # Visualizing distribution for income vs age
fig=plt.figure(figsize=(8,4))
for x in ['<=50K','>50K']:
    df['age'][df['income']==x].plot(kind='kde')
    
plt.title('Income vs Age Density Distribution')
plt.legend(('<=50K','>50K'))
plt.xlabel('Age')
# plt.savefig('./plots/income_vs_age.png')
plt.show()


# Label Encoding categorical features

le = LabelEncoder()
df['age'] = le.fit_transform(df['age'])
df['workclass'] = le.fit_transform(df['workclass'])
df['fnlwgt'] = le.fit_transform(df['fnlwgt'])
df['education'] = le.fit_transform(df['education'])
df['educational-num'] = le.fit_transform(df['educational-num'])
df['marital-status'] = le.fit_transform(df['marital-status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['gender'] = le.fit_transform(df['gender'])
df['capital-gain'] = le.fit_transform(df['capital-gain'])
df['capital-loss'] = le.fit_transform(df['capital-loss'])
df['hours-per-week'] = le.fit_transform(df['hours-per-week'])
df['native-country'] = le.fit_transform(df['native-country'])

df['income'] = le.fit_transform(df['income'])

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot = True, ax=ax)
plt.show()

# One-Hot Encoding categorical features

df['age'] = df['age'].astype('category')
df['workclass'] = df['workclass'].astype('category')
df['fnlwgt'] = df['fnlwgt'].astype('category')
df['education'] = df['education'].astype('category')
df['educational-num'] = df['educational-num'].astype('category')
df['marital-status'] = df['marital-status'].astype('category')
df['occupation'] = df['occupation'].astype('category')
df['relationship'] = df['relationship'].astype('category')
df['race'] = df['race'].astype('category')
df['gender'] = df['gender'].astype('category')
df['capital-gain'] = df['capital-gain'].astype('category')
df['capital-loss'] = df['capital-loss'].astype('category')
df['hours-per-week'] = df['hours-per-week'].astype('category')
df['native-country'] = df['native-country'].astype('category')

df = pd.get_dummies(df, columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

# X = features & y = Target class

X = df.drop(['income'], axis=1)
y = df['income']

# Normalizing the all the features

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# Building our model with K-fold validation and GridSearch to find the best parameters

# Defining all the parameters
params = {
    'penalty': ['l1','l2'],
    'C': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
}

# Building model
logreg = LogisticRegression(solver='liblinear')

# Parameter estimating using GridSearch
grid = GridSearchCV(logreg, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

# Fitting the model
grid.fit(X_train, y_train) # Takes a while  

knn_grid_val_score = grid.best_score_
print('Best Score:', knn_grid_val_score)
print('Best Params:', grid.best_params_)
print('Best Estimator:', grid.best_estimator_)

# Using the best parameters from the grid-search and predicting on test feature dataset(X_test)

knn_grid= grid.best_estimator_
y_pred = knn_grid.predict(X_test)

# Confusion matrix

confusion_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred), columns=["Predicted P", "Predicted N"], index=["Actual P","Actual N"] )
print(confusion_matrix)

# Calculating metrics

logreg_grid_score = accuracy_score(y_test, y_pred)
print('Model Accuracy:', logreg_grid_score)
print('Classification Report:\n', classification_report(y_test, y_pred))