import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_score


# Features that matter
# Pclass 

def correlation(df):
    if(False):
        dup = df['Pclass'].value_counts()
        print(dup)
        for key, val in dup.iteritems():
            print('Pclass:' + str(key), df[df['Pclass']==key]['Survived'].mean())

    if(False):
        df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)

        dup = df['Embarked'].value_counts()
        print(dup)
        for key, val in dup.iteritems():
            print('Embarked:' + str(key), df[df['Embarked']==key]['Survived'].mean())


    if(True):
        pd.options.mode.chained_assignment = None  # default='warn'
        mean = df['Age'].mean()
        std = df['Age'].std()
        for index, row in df.iterrows():
            if pd.isnull(row['Age']):
                df['Age'][index] = random.randint(math.floor(mean-std), math.ceil(mean+std))

        age = []
        fare = []
        gender = []
        embarked = []
        for index, row in df.iterrows():
            # Transform Fare to FareBand feature
            if row['Age'] > -0.001 and row['Age'] <= 20.125:
                age.append(0)
            elif row['Age'] > 20.125 and row['Age'] <= 28:
                age.append(1)
            elif row['Age'] > 28 and row['Age'] <= 38:
                age.append(2)
            elif row['Age'] > 38 and row['Age'] < 80.001:
                age.append(3)

            if row['Fare'] > -0.001 and row['Fare'] <= 7.91:
                fare.append(0)
            elif row['Fare'] > 7.91 and row['Fare'] <= 14.454:
                fare.append(1)
            elif row['Fare'] > 14.454 and row['Fare'] <= 31.0:
                fare.append(2)
            elif row['Fare'] > 31.0 and row['Fare'] < 512.33:
                fare.append(3)
            if row['Sex'] == 'female':
                gender.append(1)
            else:
                gender.append(0)
            if row['Embarked'] == 'S':
                embarked.append(0)
            elif row['Embarked'] == 'Q':
                embarked.append(1)
            else:
                embarked.append(2)

        df['AgeBand'] = age
        df['FareBand'] = fare
        df['Gender'] = gender
        df['EmbarkBand'] = embarked

        counts = df['AgeBand'].value_counts().sort_index()
        print('AgeBand groups:')
        print(counts)
        print('Correlation for age and sex:')
        for i in (0,1,2,3):
            print('AgeBand = %d | Fare = 0:\t' % i, df[(df['AgeBand']==i)&(df['FareBand']==0)]['Survived'].mean())
            print('AgeBand = %d | Fare = 1:\t' % i, df[(df['AgeBand']==i)&(df['FareBand']==1)]['Survived'].mean())
            print('AgeBand = %d | Fare = 2:\t' % i, df[(df['AgeBand']==i)&(df['FareBand']==2)]['Survived'].mean())
            print('AgeBand = %d | Fare = 3:\t' % i, df[(df['AgeBand']==i)&(df['FareBand']==3)]['Survived'].mean())
            print()
        for i in (0,1,2,3):
            print('AgeBand = %d:\t' % i, df[(df['AgeBand']==i)]['Survived'].mean())
        print()
        for i in (0,1,2,3):
            print('FareBand = %d:\t' % i, df[(df['FareBand']==i)]['Survived'].mean())
        print()
        for i in (0,1):
            print('Gender = %d:\t' % i, df[(df['Gender']==i)]['Survived'].mean())
        print()
        for i in (1,2,3):
            print('Pclass = %d:\t' % i, df[(df['Pclass']==i)]['Survived'].mean())
        print()
        SibSp_counts = df['SibSp'].value_counts().sort_index()
        for key, val in SibSp_counts.iteritems():
            print('SibSp = %d:\t' % key, df[(df['SibSp']==key)]['Survived'].mean())
        print()
        suv = df['Survived'].value_counts().sort_index()
        print(suv)
        embark_counts = df['EmbarkBand'].value_counts().sort_index()
        for key, val in embark_counts.iteritems():
            print('EmbarkBand = %d:\t' % key, df[(df['EmbarkBand']==key)]['Survived'].mean())
        print()
        Pclass = df['Pclass'].value_counts().sort_index()
        print(Pclass)
        for key, val in Pclass.iteritems():
            print('Pclass:' + str(key), df[df['Pclass']==key]['Survived'].mean())
        print()


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
    clf_decision = DecisionTreeClassifier(criterion='gini',random_state=30,max_depth=5,max_leaf_nodes=12)
    clf_decision.fit(train_df[['Gender','FareBand','EmbarkedBand','Pclass','Age', 'SibSp']], train_df['Survived'])

    clf_forest = RandomForestClassifier(criterion='gini',random_state=30,max_depth=5,max_leaf_nodes=12)
    clf_forest.fit(train_df[['Gender','FareBand','EmbarkedBand','Pclass','Age', 'SibSp']], train_df['Survived'])
    return clf_decision, clf_forest

def test(clf_gini, test_df):
    pred = clf_gini.predict(test_df)
    return pred

# Load in raw data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

# Feature correlation results
correlation(train_df)

# Preprocess raw data
proc_train_df, proc_test_df = preprocess(train_df, test_df)

# Train the data for a decision tree and random forest
clf_decision, clf_forest = train(proc_train_df)

# Plot the decision tree
fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf_decision)
fig.savefig("decision_tree.png")

# Preform 5-fold cross validation for decision tree and random forest

clf_decision, clf_forest = train(proc_train_df)
scores_tree = cross_val_score(clf_decision, proc_train_df[['Gender','FareBand','EmbarkedBand','Pclass','Age','SibSp']], proc_train_df['Survived'],cv=5, scoring='f1_micro')
scores_forest = cross_val_score(clf_forest, proc_train_df[['Gender','FareBand','EmbarkedBand','Pclass','Age','SibSp']], proc_train_df['Survived'],cv=5, scoring='f1_micro')

# Print the scores
print('Decision Tree:', scores_tree.mean())
print('Random Forest:', scores_forest.mean())
print()
