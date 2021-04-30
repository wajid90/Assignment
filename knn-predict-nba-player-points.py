#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install pandas ')
import pandas as pd 
with open("nba_2013.csv", 'r') as csvfile:
    nba = pd.read_csv(csvfile)
print(nba.columns.values)


# ### Check for null values 

# In[3]:


nba.isnull().any()


# ### fillna with series mean

# In[9]:


nba["fg."].fillna(nba["fg."].mean(),inplace=True)
nba["x2p."].fillna(nba["x2p."].mean(),inplace=True)
nba["efg."].fillna(nba["efg."].mean(),inplace=True)
nba["x3p."].fillna(nba["x3p."].mean(),inplace=True)
nba["ft."].fillna(nba["ft."].mean(),inplace=True)
nba


# ### Select only the numeric columns from the dataset

# In[8]:


distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']
nba_numeric = nba[distance_columns]
nba_numeric


# ### Normalize all of the numeric columns

# In[11]:


nba_normalized = nba_numeric.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
nba_normalized


# ### Categorical Columns

# In[14]:


nba_category = nba[['player', 'bref_team_id', 'season']]
nba_category


# In[20]:


nba = pd.concat([nba_category, nba_normalized], axis=1)

from sklearn.model_selection import train_test_split

x_columns = nba[['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']]


y_column = nba["pts"]

x_train, x_test, y_train, y_test = train_test_split(x_columns, y_column, test_size=0.3, random_state=0)


# In[21]:


get_ipython().system('pip install  scikit-learn')
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

for k in range(10):
    k_value = k + 1
    knn = KNeighborsRegressor(n_neighbors = k_value)
    knn.fit(x_train, y_train) 
    y_pred = knn.predict(x_test)
    print ("Regression score is:",format(metrics.r2_score(y_test, y_pred),'.4f'), "for k_value:", k_value)


# ### K=8, as it gives us the highest prediction score.

# In[22]:


knn = KNeighborsRegressor(n_neighbors = 8)
knn.fit(x_train, y_train) 
y_pred = knn.predict(x_test)
print ("Mean Squared Error is:", format(metrics.mean_squared_error(y_test, y_pred), '.7f'))
print ("Regression score is:", format(metrics.r2_score(y_test, y_pred),'.4f'))


# In[13]:


Test_With_Predicted = pd.DataFrame({'Actual Points': y_test.tolist(), 'Predicted Points': y_pred.tolist()})

Test_With_Predicted

