# Binary Classification for Adult Income Prediction Model

## Overview
This script implements a machine learning pipeline to predict income levels based on demographic data from the Adult Income dataset. It employs data preprocessing, visualization, feature encoding, model training, and evaluation using Logistic Regression.

## Requirements
Ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The dataset used in this script is `adult_data.csv`. It contains demographic information such as age, work class, education, and more, along with a target variable indicating income level.

## Steps in the Script

1. **Import Libraries**: Essential libraries are imported for data manipulation, visualization, and machine learning.
2. **Load Data**: The dataset is read into a pandas DataFrame using the following command:
    ```python
    df = pd.read_csv("./data/adult_data.csv")
    ```
3. **Data Cleanup**:
    * Missing data is identified and handled by dropping rows with any missing values:
       ```python
       df = df.dropna(how='any', axis=0)
       ```
    * Unnamed columns are removed:
       ```python
       df = df.dropna(how='any', axis=0)
       ```
4. **Data Visualization**:
    * Visualizing number of people in each income status:
       ```python
       sns.set_theme(style="darkgrid")
       ax = sns.countplot(x="income", data=df, palette=sns.xkcd_palette(["azure", "light red"]))
       plt.xlabel('Income')
       plt.ylabel('Count')
       plt.show()
       ```
    * Visualizing distribution for income vs age:
       ```python
       fig=plt.figure(figsize=(8,4))
       for x in ['<=50K','>50K']:
           df['age'][df['income']==x].plot(kind='kde')
       plt.title('Income vs Age Density Distribution')
       plt.legend(('<=50K','>50K'))
       plt.xlabel('Age')
       ```
5. **Modeling**: Build model based on logistic regression
    ```python
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Building model
    logreg = LogisticRegression(solver='liblinear')

    # Parameter estimating using GridSearch
    grid = GridSearchCV(logreg, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)

    # Fitting the model
    grid.fit(X_train, y_train) # Takes a while ~10 min for me
    ```
## Results

```
Best Score: 0.8543097407147862
Best Params: {'C': 3, 'penalty': 'l1'}
Best Estimator: LogisticRegression(C=3, penalty='l1', solver='liblinear')
```

Confusion Matrix:
```
            Predicted P  Predicted N
Actual P         1517           50
Actual N          246          226
```

Model Accuracy: 
```
Model Accuracy: 0.8548307994114762
```

Classification Report:
```
               precision    recall  f1-score   support

           0       0.86      0.97      0.91      1567
           1       0.82      0.48      0.60       472

    accuracy                           0.85      2039
   macro avg       0.84      0.72      0.76      2039
weighted avg       0.85      0.85      0.84      2039
```
