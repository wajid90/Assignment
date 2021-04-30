#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

import pickle


# In[2]:


cnx = sqlite3.connect('database.sqlite')


# In[3]:


dd = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", cnx)


# In[4]:


print(dd)


# In[5]:


df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[6]:


df.head()


# In[7]:


target = df.pop('overall_rating')


# In[8]:


df.shape


# In[9]:


target.head()


# ## Imputing target funtion :

# In[10]:


target.isnull().values.sum()


# there are 836 missing value present in target function.

# In[11]:


target.describe()


# In[12]:


plt.hist(target, 30, range=(33, 94))


# almost normal distribution so we can impute mean value for missing value in target.

# In[13]:


y = target.fillna(target.mean())


# In[14]:


y.isnull().values.any()


# ## Data Exploration :

# In[15]:


df.columns


# In[16]:


for col in df.columns:
    unique_cat = len(df[col].unique())
    print("{col}--> {unique_cat}..{typ}".format(col=col, unique_cat=unique_cat, typ=df[col].dtype))


# we can see only four features have the type 'object'. here the feature named 'date' has no significance in this problem so can ignore it and perform one hot encoding on the rest of 3 features.

# In[17]:


dummy_df = pd.get_dummies(df, columns=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate'])
dummy_df.head()


# In[18]:


X = dummy_df.drop(['id', 'date'], axis=1)


# ***
# ## Feature selection :

# * As tree model doesn't gets affected by missing values present in data set. but feature selection by `SelectFromModel` can not be done on datasets that carries null value. Therefore, we should also perform imputation on dataset. 

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[20]:


#imputing null value of each column with the mean of that column
imput = Imputer()
X_train = imput.fit_transform(X_train)
X_test = imput.fit_transform(X_test)


# In[21]:


#finding feature_importance for feature selection. from it we'll be able to decide threshold value
model = XGBRegressor()
model.fit(X_train, y_train)
print(model.feature_importances_)


# In[22]:


selection = SelectFromModel(model, threshold=0.01, prefit=True)

select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)


# In[23]:


select_X_train.shape


# ## Training different models :

# ### 1. Linear Regression :

# In[24]:


pipe = make_pipeline(StandardScaler(),             #preprocessing(standard scalling)
                     LinearRegression())           #estimator(linear regression)

cv = ShuffleSplit(random_state=0)   #defining type of cross_validation(shuffle spliting)

param_grid = {'linearregression__n_jobs': [-1]}     #parameters for model tunning

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


# In[25]:


grid.fit(select_X_train, y_train)          #training 


# In[26]:


grid.best_params_


# In[27]:


lin_reg = pickle.dumps(grid)


# ### 2. Decision Tree :

# In[30]:


pipe = make_pipeline(StandardScaler(),                  #preprocessing
                     DecisionTreeRegressor(criterion='mse', random_state=0))          #estimator

cv = ShuffleSplit(n_splits=10, random_state=42)        #cross validation

param_grid = {'decisiontreeregressor__max_depth': [3, 5, 7, 9, 13]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


# In[31]:


grid.fit(select_X_train, y_train)          #training 


# In[32]:


grid.best_params_


# In[33]:


Dectree_reg = pickle.dumps(grid)


# ### 3. Ranom Forest :

# In[36]:


pipe = make_pipeline(StandardScaler(),
                     RandomForestRegressor(n_estimators=500, random_state=123))

cv = ShuffleSplit(test_size=0.2, random_state=0)

param_grid = {'randomforestregressor__max_features':['sqrt', 'log2', 10],
              'randomforestregressor__max_depth':[9, 11, 13]}                 

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


# In[37]:


grid.fit(select_X_train, y_train)          #training 


# In[38]:


grid.best_params_


# In[39]:


Randfor_reg = pickle.dumps(grid)


# ### 4. Xgboost regressor :

# In[42]:


pipe = make_pipeline(StandardScaler(),
                     XGBRegressor(n_estimators= 500, random_state=42))

cv = ShuffleSplit(n_splits=10, random_state=0)

param_grid = {'xgbregressor__max_depth': [5, 7],
              'xgbregressor__learning_rate': [0.1, 0.3]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs= -1)


# In[43]:


grid.fit(select_X_train, y_train)


# In[44]:


grid.best_params_


# In[45]:


xgbreg = pickle.dumps(grid)


# ## <u>Comparision between different models</u> :

# In[48]:


lin_reg = pickle.loads(lin_reg)
Dectree_reg = pickle.loads(Dectree_reg)
Randfor_reg = pickle.loads(Randfor_reg)
xgbreg = pickle.loads(xgbreg)


# In[49]:


print("""Linear Regressor accuracy is {lin}
DecisionTree Regressor accuracy is {Dec}
RandomForest regressor accuracy is {ran}
XGBoost regressor accuracy is {xgb}""".format(lin=lin_reg.score(select_X_test, y_test),
                                                       Dec=Dectree_reg.score(select_X_test, y_test),
                                                       ran=Randfor_reg.score(select_X_test, y_test),
                                                       xgb=xgbreg.score(select_X_test, y_test)))


# By accuracy comparision performed above we can say hear that XGBoost regressor gives better result than any other model. and it can predict the target function with approx 98% accuracy.

# In[ ]:




