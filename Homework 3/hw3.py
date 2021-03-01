import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Preprocess both the test and train data frames
def preprocess(df_train, df_test):
    # Replace null test fare with the mode of train fare
    df_test['Fare'].fillna(df_train['Fare'].mode()[0], inplace=True)
    df_test['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(), inplace=True)
    df_train['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(), inplace=True)
    mean = df_train['Age'].mean()
    std = df_train['Age'].std()
    for index, row in df_train.iterrows():
        if pd.isnull(row['Age']):
            df_train.at[index,'Age'] = random.randint(math.floor(mean-std), math.ceil(mean+std))

    # Peform feature transformation
    for df in (df_train, df_test):    
        gender = []
        fare = []
        embarked = []
        for index, row in df.iterrows():
            # Trasform sex to a gender feature    
            if row['Sex'] == 'female':
                gender.append(1)
            else:
                gender.append(0)

            # Transform Fare to FareBand feature
            if row['Fare'] > -0.001 and row['Fare'] <= 7.91:
                fare.append(0)
            elif row['Fare'] > 7.91 and row['Fare'] <= 14.454:
                fare.append(1)
            elif row['Fare'] > 14.454 and row['Fare'] <= 31.0:
                fare.append(2)
            elif row['Fare'] > 31.0 and row['Fare'] < 512.33:
                fare.append(3)
            
            if row['Embarked'] == 'S':
                embarked.append(0)
            elif row['Embarked'] == 'Q':
                embarked.append(1)
            else:
                embarked.append(2)

        df['Gender'] = gender
        df['FareBand'] = fare
        df['EmbarkedBand'] = embarked

    return df_train[['Survived','Gender','FareBand','EmbarkedBand','Pclass','Age', 'SibSp']], df_test[['Gender','FareBand','EmbarkedBand','Pclass','Age', 'SibSp']]

def train(train_df):
    clf_svm = []
    clf_svm.append(svm.SVC(kernel='linear'))
    clf_svm.append(svm.SVC(kernel='poly', degree=2))
    clf_svm.append(svm.SVC(kernel='rbf'))
    for i in range(len(clf_svm)):
        clf_svm[i].fit(train_df.loc[:, train_df.columns != 'Survived'], train_df['Survived'])

    return clf_svm


# Load in raw data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

# Preprocess raw data
proc_train_df, proc_test_df = preprocess(train_df, test_df)

# Train the data for SVM
clf_svm = train(proc_train_df)

# Preform 5-fold cross validation for each SVM model
for clf in clf_svm:
    score = cross_val_score(clf, proc_train_df.loc[:, proc_train_df.columns != 'Survived'], proc_train_df['Survived'],cv=5)
    title = 'SVM with ' + str(clf.kernel)
    if clf.kernel == 'poly':
            title = title + ' of degree ' + str(clf.degree)
    title = title + ' kernel'
    print(title)
    print('Scores:', score)
    print('Average:', score.mean())
    print()
