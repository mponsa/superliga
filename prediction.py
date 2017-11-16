# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:02:09 2017

@author: mponsa
"""
#---------------------------------------------------------------------------------------------------------------------------------#
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
#displayd data
from IPython.display import display
#---------------------------------------------------------------------------------------------------------------------------------#
# Importing the dataset
dataset = pd.read_csv('ARG.csv', sep=",") 
#---------------------------------------------------------------------------------------------------------------------------------#
#Traducir Hora a Mañana = 0, Tarde = 1 , Noche = 2. Tener en cuenta que el horario es el de UK UTC 00:00.  Hay que restarle 3 Horas.
#Treshold / 08:00 a 13:00 Mañana - 14 a 19 Tarde - 20 a 24 Noche /
#Obtengo la hora del partido
dataset['Time'] = dataset['Time'].str[:2]
dataset['Time'] = dataset['Time'].astype(str).astype(int)
dataset['Time'] = dataset['Time'] - 3
dataset['Time'] = dataset['Time'].where(dataset['Time'] > 0 ,dataset['Time'] + 24)

for index in range(0,dataset['Time'].size):
    if 8<=dataset['Time'][index]<=13:
        dataset['Time'].loc[index] = 0
    if 14<=dataset['Time'][index]<=19:
        dataset['Time'].loc[index] = 1
    if 20<=dataset['Time'][index]<=24:
        dataset['Time'].loc[index] = 2
        


#---------------------------------------------------------------------------------------------------------------------------------#
#Treating Season column
seasons = dataset['Season'].unique()
keys = list(range(0,seasons.size))
#Create dictionary for mapping function
dict_seasons = dict(zip(seasons,keys))
dict_seasons_inv = dict(zip(keys,seasons))

dataset['Season'] = dataset['Season'].map(dict_seasons)
#---------------------------------------------------------------------------------------------------------------------------------#
#Treating Date Column
dataset['Date'] = pd.to_datetime(dataset['Date'])
#---------------------------------------------------------------------------------------------------------------------------------#
#Creating Accumulated values by team.
dataset['ColOfIndex'] = dataset.index

#Valores acumulados para cada equipo
dataset['GFL'] = dataset.groupby('Home').HG.cumsum()
dataset['GCL'] = dataset.groupby('Home').AG.cumsum()
dataset['GFLS'] = dataset.groupby(['Home','Season']).HG.cumsum()
dataset['GCLS'] = dataset.groupby(['Home','Season']).AG.cumsum()
dataset['GFV'] = dataset.groupby('Away').AG.cumsum()
dataset['GCV'] = dataset.groupby('Away').HG.cumsum()
dataset['GFVS'] = dataset.groupby(['Away','Season']).HG.cumsum()
dataset['GCVS'] = dataset.groupby(['Away','Season']).AG.cumsum()

#Esto define con cuantos goles llegan a cada partido.
dataset['GFL'] = dataset['GFL'] - dataset['HG']
dataset['GCL'] = dataset['GCL'] - dataset['AG']
dataset['GFLS'] = dataset['GFLS'] - dataset['HG']
dataset['GCLS'] = dataset['GCLS'] - dataset['AG']
dataset['GFV'] = dataset['GFV'] - dataset['AG']
dataset['GCV'] = dataset['GCV'] - dataset['HG']
dataset['GFVS'] = dataset['GFVS'] - dataset['AG']
dataset['GCVS'] = dataset['GCVS'] - dataset['HG']


#dataset = dataset.sort_values(['Home','ColOfIndex'])


dataset = dataset.drop(['ColOfIndex'],1)
#---------------------------------------------------------------------------------------------------------------------------------#    
#Treating Home and Away Columns
teams = dataset['Home'].unique()
keys = list(range(0,teams.size))
#Create dictionary for mapping function
dict_teams = dict(zip(teams,keys))
dict_teams_inv = dict(zip(keys,teams))


dataset['Home'] = dataset['Home'].map(dict_teams)
dataset['Away'] = dataset['Away'].map(dict_teams)
#---------------------------------------------------------------------------------------------------------------------------------#
#Calculo de la media por equipo, para las columnas de Apuestas vacias en los datos nuevos.
homecolumns = [['PH','MaxH','AvgH']]
awaycolumns = [['PA','MaxA','AvgA']]
drawcolumns = [['PD','MaxD','AvgD']]



for col in homecolumns:
    dataset[col] = dataset.groupby(['Home'])[col].ffill()
for col in awaycolumns:
    dataset[col] = dataset.groupby(['Away'])[col].ffill()
for col in drawcolumns:     
    dataset[col] = dataset.groupby(['Home'])[col].ffill()



#---------------------------------------------------------------------------------------------------------------------------------#
#Dataset treatment finalized.
#---------------------------------------------------------------------------------------------------------------------------------#
#Plot some values 
plt.hist(dataset['HG'], bins = 6, color = 'red')
plt.xlabel("Home Goals distribution")
plt.ylabel("Match count / Frecuency")

plt.show()


plt.hist(dataset['AG'],bins = 6, color = 'blue')
plt.xlabel("Away Goals distribution")
plt.ylabel("Match Count / Frecuency")

plt.show()

plt.hist(dataset['HG'][dataset['Res'] == 'A'], bins = 6, color = 'blue')
plt.xlabel("Away Goals distribution")
plt.ylabel("Match Count / Frecuency")

plt.show()

plt.hist(dataset['Time'][dataset['Res'] == 'H'],bins = 3, color = 'blue')
plt.xlabel("Time distribution")
plt.ylabel("Match Count / Frecuency")

plt.show()

from pandas.tools.plotting import scatter_matrix

scatter_matrix(dataset[['PH','PD','PA','GFL','GFV','GCL','GCV']], figsize=(10,10))


#---------------------------------------------------------------------------------------------------------------------------------#
#Dataset Visualization finalized
#---------------------------------------------------------------------------------------------------------------------------------#
#Separate Independent from dependant variables.
X = dataset.drop(['Res'],1)
Y = dataset['Res']

dict_res = {'H': 0, 'A': 1, 'D': 2}
dict_res_inv = {0: 'H', 1: 'A',2: 'D' }


Y = Y.map(dict_res).astype("category").cat.codes
#---------------------------------------------------------------------------------------------------------------------------------#
#Drop unmeaningful values Like 'Country'and League. 'Prevent dataset of overfitting' deleting Home goals and Against Goals..
X = X.drop(['Country','League','HG','AG'],1)
#---------------------------------------------------------------------------------------------------------------------------------#
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer

X_copy = X.drop(['Date'],1)
#Data Standarization
imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)
columns = [['PD','PH','PA','MaxH','MaxA','MaxD','AvgH','AvgA','AvgD']]


for col in columns:
   imputer = imputer.fit(X_copy[col])
   X_copy[col] = imputer.transform(X_copy[col])
    

#Data Standarization.
myscaler = preprocessing.StandardScaler()
myscaler.fit(X_copy)
X_copy = myscaler.transform(X_copy)
#---------------------------------------------------------------------------------------------------------------------------------#
#Data standarization finished.
#---------------------------------------------------------------------------------------------------------------------------------#
#from sklearn.cross_validation import train_test_split
## Shuffle and split the dataset into training and testing set.
#X_train, X_test, y_train, y_test = train_test_split(X_copy, Y, 
#                                                    test_size = .20,
#                                                    random_state = 0,
#  
from sklearn import model_selection
                                                  
X_train = X_copy[:(index-400)]
y_train = Y[:(index-400)]

X_test = X_copy[-14:]
y_test = Y[-14:]




#Finding optimal parameters
seed = 123
 
# RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
clf_rf = RandomForestClassifier(n_jobs=-1, oob_score = True, max_depth=10, max_features='sqrt', min_samples_leaf = 1) 

#Get the best number of estimators.
## Range of `n_estimators` values to explore.
#n_estim = list(range(10,100))
# 
#cv_scores = []
# 
#for i in n_estim:
#    clf_rf.set_params(n_estimators=i)
#    kfold = model_selection.KFold(n_splits=10, random_state=seed)
#    scores = model_selection.cross_val_score(clf_rf, X_train, y_train, cv=kfold, scoring='accuracy')
#    cv_scores.append(scores.mean()*100)
#    
#optimal_n_estim = n_estim[cv_scores.index(max(cv_scores))]
#print ("The optimal number of estimators is %d with %0.1f%%" % (optimal_n_estim, cv_scores[optimal_n_estim]))
# 
#plt.plot(n_estim, cv_scores)
#plt.xlabel('Number of Estimators')
#plt.ylabel('Train Accuracy')
#plt.show()

clf_rf.set_params(n_estimators = 84)
clf_rf.fit(X_train,y_train)

clf_svc = SVC(random_state = 123, kernel='rbf')
clf_svc.fit(X_train,y_train)


y_pred = clf_rf.predict(X_test)
y_pred1 = clf_svc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred)
cm_svc = confusion_matrix(y_test, y_pred1)





from sklearn.metrics import roc_auc_score

Y_query = pd.Series(y_pred).map(dict_res_inv)

dataset_query = dataset[['Home','Away']][-14:]
dataset_query['Home'] = dataset_query['Home'].map(dict_teams_inv)
dataset_query['Away'] = dataset_query['Away'].map(dict_teams_inv)
dataset_query['Real result'] = dataset['Res'][-14:]
dataset_query['Predicted result'] = Y_query.values



roc_auc_score()

