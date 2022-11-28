# -*- coding: utf-8 -*-
"""
@author: Soundarya Lahari Valipe
"""

#Libraries
import numpy as np
import pandas as pd
from catboost import Pool
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

#Loading the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.head()


df_train['Train'] = 'Yes'
df_test['Train'] = 'No'

merged = pd.concat([df_train, df_test])
print(merged)

#Exploratory data analysis
merged.hist(bins = 25, figsize = (12, 12))
plt.show()

columns=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',]
q, r =divmod(len(columns), 2)
fig, ax=plt.subplots(q, 2, figsize=(18,10))
for i in range(0,len(columns)):
    q, r =divmod(i, 2)
    sns.boxplot(data=merged, x=columns[i], ax=ax[q, r])
plt.show()


#Feature engineering
#Imputation
merged.isnull().sum()

#Dealing with missing numerical data in the dataset

merged[['Age', 'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].median()

merged[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = merged[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

merged['Age'] = merged['Age'].fillna(merged['Age'].median())

#Dealing with the missing categorical data
merged['VIP'].value_counts()

merged['VIP'] = merged['VIP'].fillna(False)
merged['HomePlanet'].value_counts()

merged['HomePlanet'] = merged['HomePlanet'].fillna('Mars')
merged['Destination'].value_counts()

merged['Destination'] = merged['Destination'].fillna("'TRAPPIST-1e")
merged['CryoSleep'].value_counts()

merged['CryoSleep'] = merged['CryoSleep'].fillna(False)

merged[['Deck', 'Num', 'Side']] = merged['Cabin'].str.split('/', expand=True)
print(merged)

merged['Deck'].value_counts()

merged['Deck'] = merged['Deck'].fillna('T')

merged['Num'].value_counts()

merged['Num'] = merged['Num'].fillna('0')

merged['Side'].value_counts()

merged['Side'] = merged['Side'].fillna('P')

merged['Sum_spend'] = merged['RoomService'] + merged['FoodCourt'] + merged['ShoppingMall'] + merged['Spa'] + merged['VRDeck']
print(merged)

merged['AgeGroup'] = pd.cut(merged.Age, bins=[-1, 5, 13, 18, 60, 100], labels = ['Baby', 'Child', 'Teen', 'Adult', 'Elderly'])
print(merged)

merged = merged.drop(['Name', 'Cabin'],axis=1)
print(merged)

print(merged.isnull().sum())

#Label encoding

categorical_cols= ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side','Num', 'AgeGroup']
for i in categorical_cols:
    print(i)
    le=LabelEncoder()
    arr = np.array(merged[i]).astype(str)
    le.fit(arr)
    merged[i]=le.transform(merged[i].astype(str))
    
merged.set_index('PassengerId',inplace=True)
print(merged.head())

df_train = merged[merged['Train'] == 'Yes']
df_train.drop('Train', axis=1, inplace=True)
print(df_train)

df_test = merged[merged['Train'] == 'No']
df_test.drop('Train', axis=1, inplace=True)
print(df_test)

df_train['Transported']=df_train['Transported'].replace({True:1,False:0})
print(df_train)

#Preparation of data
X=df_train.drop('Transported',axis=1)
y = df_train['Transported']
print(X.columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


#######Model selection######
#CatBoost Classifier
from catboost import CatBoostClassifier
cat=CatBoostClassifier(iterations=1500,
                         eval_metric='Accuracy',
                        verbose=0)
cat.fit(X_train,y_train)
#Value prediction
pred = cat.predict(X_train)
pred_y = cat.predict(X_val)
print('CatBoost Classifier:')
print('Training accuracy',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))

#Gradient Boosting Classifier
gb=GradientBoostingClassifier(random_state=1,n_estimators=250,learning_rate=0.15,max_depth=3)
gb.fit(X_train,y_train)
#Value prediction
pred=gb.predict(X_train)
pred_y=gb.predict(X_val)
print('Gradient Boosting Classifier:')
print('Training accuracy',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))

#GridSearchCV
gcv=GridSearchCV(CatBoostClassifier(),param_grid={'iterations': range(200,2000,200), 'eval_metric': ['Accuracy'],'verbose':[0]},cv=3)
gcv.fit(X_train,y_train)
#Value prediction
pred=gcv.predict(X_train)
pred_y=gcv.predict(X_val)
print('GridSearchCV:')
print('Training accuracy',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))

gcv.fit(X,y)
y_pred = gcv.predict(df_test)

#Submission
submit=pd.DataFrame({'Transported':y_pred.astype(bool)},index=df_test.index)
submit.reset_index(inplace=True)
print(submit.head())

submit.to_csv('submission.csv', index=False)
