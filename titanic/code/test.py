import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')


train_and_test = [train, test]

# for dataset in train_and_test:
#     ​	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

# train.head(5)

# for dataset in train_and_test:
#     ​    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
# ​                                                 'Lady','Major', 'Rev', 'Sir'], 'Other')
# ​    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
# ​    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# ​    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



# for dataset in train_and_test:
#     ​    dataset['Title'] = dataset['Title'].astype(str)

# for dataset in train_and_test:
#     ​    dataset['Sex'] = dataset['Sex'].astype(str)


train.Embared.value_count(dropna=False)

# for dataset in train_and_test:
#     ​    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# ​    dataset['Embarked'] = dataset['Embarked'].astype(str)

for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()) # Survivied ratio about Age Band

# for dataset in train_and_test:
#     ​dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
# ​    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
# ​    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
# ​    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
# ​    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# ​    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)

print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])

# for dataset in train_and_test:
# ​    dataset['Fare'] = dataset['Fare'].fillna(13.675)

# for dataset in train_and_test:
#     ​dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
#     dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
#     dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
#     dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
#     dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
#     dataset['Fare'] = dataset['Fare'].astype(int)

# for dataset in train_and_test:
#     ​dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
# ​    dataset['Family'] = dataset['Family'].astype(int)

features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

print(train.head())
print(test.head())


train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle

train_data, train_label = shuffle(train_data, train_label, random_state = 5)

def train_and_tests(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction


log_pred = train_and_tests(LogisticRegression())
# SVM
svm_pred = train_and_tests(SVC())
#kNN
knn_pred_4 = train_and_tests(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_tests(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_tests(GaussianNB())


submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": rf_pred})

submission.to_csv('submission_rf.csv', index=False)