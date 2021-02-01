import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def question_5(df, test_df):
    print('Question 5')

    df_null = df.isnull().sum()
    print(df_null)

    test_null = test_df.isnull().sum()
    print(test_null)

def question_7(df):
    print('Question 7')
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    for feature in features:
        print('For feature: ', feature)
        print('Count: ', df[feature].count())
        print('Mean: ', df[feature].mean())
        print('STD: ', df[feature].std())
        print('Percentile: ', df[feature].quantile([0.25,0.5,0.75]))
        print('Max: ', df[feature].max())
        print()


def question_8(df):
    print('Question 8')
    features = ['PassengerId', 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    for feature in features:
        print('For feature: ', feature)
        print('Count: ', df[feature].count())
        print(df[feature].value_counts())
        print()
    
    return
    

def question_9_10(train_df):
    pclass = 0
    pclass_count = 0
    sexf = 0
    sexf_count = 0
    sexm = 0
    sexm_count = 0

    for index, row in train_df.iterrows():
        if row['Pclass'] == 1:
            pclass = pclass + row['Survived']
            pclass_count = pclass_count + 1
        if row['Sex'] == 'female':
            sexf = sexf + row['Survived']
            sexf_count = sexf_count + 1
        if row['Sex'] == 'male':
            sexm = sexm + row['Survived']
            sexm_count = sexm_count + 1

    print("Question 9:")
    print("Total count of Pclass = 1: " + str(pclass_count))
    print("Average survived: " + str(pclass / pclass_count))

    print("Question 10")
    print("Average female: " + str(sexf/sexf_count))
    print("Average male: " + str(sexm/sexm_count))

def question_11(train_df):
    print("Question 11")
    train_df.dropna(subset=['Age'], inplace=True)
    df1 = train_df[train_df['Survived']==0]
    df2 = train_df[train_df['Survived']==1]

    fig, axes = plt.subplots(ncols=2, sharey=True)

    axes[0].hist(df1['Age'],bins=20)
    axes[0].set_xlabel('Age')
    axes[0].set_title('Survived = 0')
    axes[0].set_xlim([0,80])

    axes[1].hist(df2['Age'],bins=20)
    axes[1].set_xlabel('Age')
    axes[1].set_title('Survived = 1')
    axes[1].set_xlim([0,80])

    plt.show()

def question_12(train_df):
    print('Question 12')
    train_df.dropna(subset=['Age','Pclass','Survived'], inplace=True)
    df1 = []
    df1.append(train_df[(train_df['Survived']==0) & (train_df['Pclass']==1)])
    df1.append(train_df[(train_df['Survived']==0) & (train_df['Pclass']==2)])
    df1.append(train_df[(train_df['Survived']==0) & (train_df['Pclass']==3)])
    df2 = []
    df2.append(train_df[(train_df['Survived']==1) & (train_df['Pclass']==1)])
    df2.append(train_df[(train_df['Survived']==1) & (train_df['Pclass']==2)])
    df2.append(train_df[(train_df['Survived']==1) & (train_df['Pclass']==3)])

    fig, axes = plt.subplots(ncols=2,nrows=3,sharey=True,sharex=True)

    for i in range(3):
        axes[i][0].hist(df1[i]['Age'],bins=20)
        axes[i][0].set_title('Pclass = ' + str(i+1) +' | Survived = 0')
        axes[i][0].set_xlim([0,80])

        axes[i][1].hist(df2[i]['Age'],bins=20)
        axes[i][1].set_title('Pclass = ' + str(i+1) +' | Survived = 1')
        axes[i][1].set_xlim([0,80])

    axes[2][0].set_xlabel('Age')
    axes[2][1].set_xlabel('Age')

    plt.show()

def question_13(train_df):
    print('Question 13')

    labels = ['C','Q','S']

    fig, axes = plt.subplots(ncols=2,nrows=3,sharey=True,sharex=True)
    filtered = train_df.groupby(['Sex','Survived','Embarked'])['Fare'].mean()

    # Legend for the filtered df
    # 0  - survived = 0 | Embarked = C female || 6  - survived = 0 | Embarked = C male 
    # 1  - survived = 0 | Embarked = Q female || 7  - survived = 0 | Embarked = Q male 
    # 2  - survived = 0 | Embarked = S female || 8  - survived = 0 | Embarked = S male 
    # 3  - survived = 1 | Embarked = C female || 9  - survived = 1 | Embarked = C male 
    # 4  - survived = 1 | Embarked = Q female || 10 - survived = 1 | Embarked = Q male 
    # 5  - survived = 1 | Embarked = S female || 11 - survived = 1 | Embarked = S male 
    
    for i in range(3):
        axes[i][0].bar(['female','male'], [filtered[i],filtered[i+6]])
        axes[i][0].set_title('Embarked = ' + labels[i] +' | Survived = 0',fontsize=10)

        axes[i][1].bar(['female','male'], [filtered[i+3],filtered[i+9]])
        axes[i][1].set_title('Embarked = ' + labels[i] +' | Survived = 1',fontsize=10)

    axes[0][0].set_ylabel('Fare')
    axes[1][0].set_ylabel('Fare')
    axes[2][0].set_ylabel('Fare')
    axes[2][0].set_xlabel('Sex')
    axes[2][1].set_xlabel('Sex')

    plt.show()

def question_14(df):
    print('Question 14')

    dup_tickets = df['Ticket'].value_counts()
    tickets = 0
    print(dup_tickets)
    for key, val in dup_tickets.iteritems():
        if val == 1:
            break       

        print('Ticket:' + str(key), df[df['Ticket']==key]['Survived'].mean())
        tickets = tickets + val
    print('Total tickets: ' + str(len(train_df['Ticket'])))
    print('Number of Duplicates: ' + str(tickets), 'Rate: ' + str(tickets/len(train_df['Ticket'])))


def question_15(df):
    print('Question 15')
    print('Null Values: ', df['Cabin'].isnull().sum())
    print('Total Values:', len(df['PassengerId']))

def question_16(df):
    print('Question 16')
    gender = []
    for index, row in df.iterrows():
        if row['Sex'] == 'female':
            gender.append(1)
        else:
            gender.append(0)
    df['Gender'] = gender

    print(df[['Sex','Gender']])

def question_17(df):
    print('Question 17')
    print('Before Replacement: ')
    print(df[['PassengerId','Age']])

    mean = df['Age'].mean()
    std = df['Age'].std()
    print('Mean:', mean, 'STD: ', std)
    for index, row in df.iterrows():
        if pd.isnull(row['Age']):
            df['Age'][index] = random.randint(math.floor(mean-std), math.ceil(mean+std))

    print('After Replacement')
    print(df[['PassengerId','Age']])

def question_18(df):
    print('Question 18')
    labels = ['S','Q','C']
    null = df[df['Embarked'].isnull()].index.tolist()

    print('Before replacement...')
    for i in null:
        print(df.loc[[i]])
    print()

    df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)

    print('After replacement...')
    for i in null:
        print(df.loc[[i]])

def question_19(df, df_test):
    print('Question 19')
    print('Most common Fare by mode:', df['Fare'].mode()[0])

    pd.to_numeric(df_test['Fare'])
    print((df_test['Fare']))
    labels = ['S','Q','C']
    null = df_test[df_test['Fare'].isnull()].index.tolist()

    print('Before replacement...')
    for i in null:
        print(df_test.loc[[i]])
    print()

    df_test['Fare'].fillna(df['Fare'].mode()[0], inplace=True)

    print('After replacement...')
    for i in null:
        print(df_test.loc[[i]])

def question_20(df):
    print('Question 20')
    for index, row in df.iterrows():
        if row['Fare'] > -0.001 and row['Fare'] <= 7.91:
            df['Fare'][index] = 0
        elif row['Fare'] > 7.91 and row['Fare'] <= 14.454:
            df['Fare'][index] = 1
        elif row['Fare'] > 14.454 and row['Fare'] <= 31.0:
            df['Fare'][index] = 2
        elif row['Fare'] > 31.0 and row['Fare'] < 512.33:
            df['Fare'][index] = 3
    
    df['Fare'] = df['Fare'].astype('Int64')

    print(df)

# Avoid replacement warnings
pd.options.mode.chained_assignment = None  # default='warn'

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df.copy(), test_df.copy()]

question_5(train_df.copy(), test_df.copy())
question_7(train_df.copy())
question_8(train_df.copy())
question_9_10(train_df.copy())
question_11(train_df.copy())
question_12(train_df.copy())
question_13(train_df.copy())
question_14(train_df.copy())
question_15(pd.concat(combine))
question_16(train_df.copy())
question_17(train_df.copy())
question_18(train_df.copy())
question_19(train_df.copy(), test_df.copy())
question_20(train_df.copy())