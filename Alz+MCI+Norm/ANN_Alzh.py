#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score


# In[2]:


data = pd.read_csv('GSE63063-GPL10558_trimmed.csv',header= 0)


# In[3]:


df = data
df.head()


# In[4]:


df = df.T


# In[5]:


datax = df.drop('Patient_barcode')
datax = datax.drop([0,2,3],axis=1)
datay = (df[1][2:])
datax.fillna(0,inplace=True)
#df.columns = df.iloc[1]
print(datax.shape)
print(datay.shape)
datax.head()


# In[8]:


datax.columns = (datax.iloc[0])
datax.drop(0,inplace=True,axis=1)
datax.drop('!series_matrix_table_end',axis = 1,inplace=True)
datax.drop('Probe_ID',inplace=True)
datax.head()
# datax.to_csv('Alzh_Features_Wrangled.csv')
# datay.to_csv('Alzh_Labels_Wrangled.csv')


# In[ ]:


from sklearn.preprocessing import StandardScaler as SS
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.pipeline import Pipeline
SS = SS()
clf = MLP()
#print(clf.get_params().keys())
pipe = Pipeline(steps=[('scaler', SS), ('MLP', clf)])
params = {'MLP__hidden_layer_sizes':list(range(1000,30000,1000)),'MLP__activation':['logistic', 'tanh', 'relu']}


grid_search = GSCV(pipe, params, cv=8, scoring='accuracy')
grid_search.fit(datax,datay)
best_act = grid_search.best_params_.get('MLP__activation')
best_hl = grid_search.best_params_.get('MLP__hidden_layer_size')
print('Best Parameters:',grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)

nested_score = cross_val_score(grid_search, datax, datay, cv=8)
print('Nested Score:',nested_score.mean())


# In[ ]:


import pickle
final_model = grid_search

filename = 'finalized_ANN_Alzh.sav'
pickle.dump(final_model, open(filename, 'wb'))


# In[ ]:




