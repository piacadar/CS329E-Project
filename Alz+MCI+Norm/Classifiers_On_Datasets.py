#!/usr/bin/env python
# coding: utf-8

# In[6]:


import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score


# In[7]:


data = pd.read_csv('Full_subset_Alzh.csv',header= 0)


# In[8]:


df = data
df.head()


# In[9]:


df = df.T


# In[10]:


datax = df.drop('Patient_barcode')
datax = datax.drop([0,2,3],axis=1)
datay = (df[1][2:])
#df.columns = df.iloc[1]
datax.drop(1,inplace=True,axis=1)

print(datax.shape)
print(datay.shape)
datax.head()


# In[12]:


datax.columns = (datax.iloc[0])
#datax.drop('!series_matrix_table_end',axis = 1,inplace=True)
datax.drop('Probe_ID',inplace=True)
datax.fillna(datax.mean(),inplace=True)
# # # datax.to_csv('Alzh_Features_Wrangled.csv')
# # # datay.to_csv('Alzh_Labels_Wrangled.csv')


# In[13]:


print(datax.shape)
print(datay.shape)


# In[77]:


# from sklearn.preprocessing import StandardScaler as SS
# from sklearn.neural_network import MLPClassifier as MLP
# from sklearn.pipeline import Pipeline
# import itertools
# SS = SS()
# clf = MLP()
# #print(clf.get_params().keys())
# pipe = Pipeline(steps=[('scaler', SS), ('MLP', clf)])
# params = {'MLP__hidden_layer_sizes': [x for x in itertools.product((50,70),repeat=2)],'MLP__activation':['logistic', 'tanh', 'relu']}


# grid_search = GSCV(pipe, params, cv=8, scoring='f1_macro')
# grid_search.fit(newx,newy)
# best_act = grid_search.best_params_.get('MLP__activation')
# best_hl = grid_search.best_params_.get('MLP__hidden_layer_size')
# print('Best Parameters:',grid_search.best_params_)
# print("F_score:", grid_search.best_score_)

# nested_score = cross_val_score(grid_search, newx, newy, cv=3)
# print('Nested Score:',nested_score.mean())


# In[18]:


# your code goes here
from sklearn.ensemble import RandomForestClassifier as RFC
clf = RFC()
params = {'max_depth':list(range(40,80)), 'min_samples_leaf':[2,3,4,5,6,7,10], 'max_features':['sqrt','log2']}
grid_search = GSCV(clf, params, cv=15, scoring='f1_macro')
grid_search.fit(datax,datay)

best_depth = grid_search.best_params_.get('max_depth')
best_msl = grid_search.best_params_.get('min_samples_leaf')
best_features = grid_search.best_params_.get('max_features')

print('Best Parameters:',grid_search.best_params_)
print("F_score:", grid_search.best_score_)


# In[19]:


nested_score = cross_val_score(grid_search, datax, datay, cv=15)
print('Nested Score:',nested_score.mean())


# In[17]:


import pickle as pickle_rick
final_model = grid_search

filename = 'finalized_RFC_Alzh.sav'
pickle_rick.dump(final_model, open(filename, 'wb'))


# In[ ]:




