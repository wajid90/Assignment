#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from xgboost.sklearn import XGBClassifier


# In[ ]:


url='./adult.data'

cols=["age","workclass","fnlwgt","education","education-num","marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week","native-country","wage_class"]

train_set=pd.read_csv(url, names=cols)

train_set.head()


# In[3]:


test_set = pd.read_csv('./adult.test',skiprows = 1, header = None, names=cols)   #coz row one is multi index 

test_set.head()


# ## EDA

# In[4]:


train_set.isnull().sum()
   # no null values in any frame


# In[5]:


train_set.hist(figsize=(10,10))
plt.show()


# In[6]:


train_set.dtypes  
#only 6 features(out of 14 are numeric)


# ##### checking uniqe values and count of them in each columns having type as object

# In[7]:



train_set.workclass.value_counts()


# In[8]:


for feature in cols:
    if train_set[feature].dtype == 'object':
        
        print('_'*10,feature, end=" ",)
        print('_'*10)
        print(train_set[feature].value_counts())
        


# **Observation:** features `Native_counrty, Occupation, workclass` having '?' as one value need to be handle

# get index of the ? in the above mentioned columns

# #### some features name containing `-` in the name remove or replace with `underscore`

# In[11]:


train_set.columns  


# ##### using regular expression

# In[12]:


import re
train_set.rename(columns=lambda name: re.sub(r"\-",'_',name), inplace=True)  


# In[13]:


filter1 = train_set["workclass"]=="?"
#index= train_set.where(filter1).index
index = train_set[train_set["native_country"]== ' ?'].index
#index.value_counts()  # to display all the indexes having value '?'
index.value_counts().sum()  # sum of those indexes 
# type(index)


# In[14]:


#now apply the filter for the '?' value in those above 3 columns

indexes_list = [] # will contain three index series of desired syntexes
def filter_questionMark(l):
    
    for item in l:
        print('_'*10,item, '_'*10)
        index = train_set[train_set[item]== ' ?'].index
        indexes_list.append(index)
        #index.value_counts()  # to display all the indexes having value '?'
        print("Total `?` in {0}: {1}".format(item, index.value_counts().sum()))  # sum of those indexes 
        print("Total % of `?` in {0}: {1}".format(item, round((index.value_counts().sum()/train_set.shape[0])*100,2)))
        
filter2= ['workclass','occupation', 'native_country']
filter_questionMark(filter2)


# **Observation:** Max % of having `?` is 5.66 we can drop and can check how much it is affedting our data

# In[15]:


print("Original Shape of Train Set:", train_set.shape)
print("Rows in Train Set:", train_set.shape[0])


# In[16]:


print('\n Drowping `?` indexes from `workclass` featutre....')
train_set= train_set.drop(index)
print("Total New rows:",train_set.shape[0])


# In[17]:


print('\n Drowping `?` indexes from `native_country` featutre....')
index1= train_set[train_set["native_country"]== ' ?'].index
train_set = train_set.drop(index1)
print("Total New rows:",train_set.shape[0])


# In[18]:


print('\n Drowping `?` indexes from `occupation` featutre....')
index2 = train_set[train_set["occupation"]== ' ?'].index
train_set = train_set.drop(index2)
print("Total New rows:",train_set.shape[0])


# In[19]:


# check for '?' again
filter_questionMark(filter2)


# **Note:** As XGBoost dnt support categorical data we need to use one hot coding tochange all object type features to numeric

# #### Applying one hot coding to all the categorical variables

# In[20]:


df1 = train_set.copy()

objectFeature = []
for i in list(df1.columns):
    if (df1[i].dtypes == 'object'):
        objectFeature.append(i)
        

df1=pd.get_dummies(df1[objectFeature[:-1]])


# In[21]:


intFeature = []
for i in list(train_set.columns):
    if (train_set[i].dtypes == 'int64'):
        intFeature.append(i)

df2 = train_set[intFeature]


# In[22]:


print(df1.shape)
print(df2.shape)


# ## Concatinating the two frames 

# In[23]:


X_train = pd.concat([df1, df2], axis=1)


# In[24]:


X_train.shape   # final dataframe to be get Trained 


# In[25]:


y=train_set[objectFeature[-1]]


# ### Using LabelEncoder over the target variable

# In[38]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# le.fit(y)
# le.classes_   # will display only two classes---> array([' <=50K', ' >50K'], dtype=object)y_train
y_train=le.fit_transform(y)




# In[40]:


y_train


# same process we can apply with the test data to prepare it for test input, secondly we have combined both and performed same operations and later divied again into train and test split
# 
# 
# i am using same from the training portion to demonstrate model

# # Applying XGB : the sklearn way

# In[51]:


from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[67]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': .5,
    'silent': True,    # would be boolean in sklearn
    'n_estimators': 100
}


# In[41]:


x_train, x_test, y_t1, y_t2 =train_test_split(X_train, y_train, test_size=.25)


# In[42]:


x_train.shape, x_test.shape, y_t1.shape, y_t2.shape


# #### Model Training : making model 1

# In[77]:


xgb = XGBClassifier(**params).fit(x_train,y_t1)


# In[65]:


y_pred= xgb.predict(x_test)


# In[68]:


accuracy_score(y_t2, y_pred)


# #### Making model no .2

# In[70]:


eval_set = [(x_train,y_t1), (x_test, y_t2)]
xgb1 = XGBClassifier(**params).fit(x_train,y_t1,
                                   early_stopping_rounds=15, 
                                   eval_metric=["error", "logloss"], 
                                   eval_set=eval_set,
                                   verbose=True)


# **n_estimators — the number of runs XGBoost will try to learn**
# 
# **learning_rate — learning speed**
# 
# **early_stopping_rounds — overfitting prevention, stop early if no improvement in learning**

# In[72]:


y_pred1=xgb1.predict(x_test)


# In[74]:


accuracy_score(y_t2, y_pred1)


# ## Ploting Classifying errors and log loss with respect to each iteration

# In[80]:


# retrieve performance metrics
results = xgb1.evals_result()
epochs = len(results['validation_0']['error'])
x = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x, results['validation_0']['logloss'], label='Train')
ax.plot(x, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.xlabel('Epochs')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x, results['validation_0']['error'], label='Train')
ax.plot(x, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.xlabel('Epochs')
plt.title('XGBoost Classification Error')
plt.show()


# In[ ]:





# ### model no. 3

# In[81]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 20,
    'learning_rate': .01,
    'silent': True,    # would be boolean in sklearn
    'n_estimators': 200
}


# In[82]:


eval_set = [(x_train,y_t1), (x_test, y_t2)]
xgb1 = XGBClassifier(**params).fit(x_train,y_t1,
                                   early_stopping_rounds=15, 
                                   eval_metric=["error", "logloss"], 
                                   eval_set=eval_set,
                                   verbose=True)


# In[83]:


#plot
# retrieve performance metrics
results = xgb1.evals_result()
epochs = len(results['validation_0']['error'])
x = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x, results['validation_0']['logloss'], label='Train')
ax.plot(x, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.xlabel('Epochs')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x, results['validation_0']['error'], label='Train')
ax.plot(x, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.xlabel('Epochs')
plt.title('XGBoost Classification Error')
plt.show()


# ###### Observation

# By adjusting parameter we can improve over accuracy,

# ## The XGBoost way

# In[86]:


#Import Xgboost
import xgboost as xgb


# In[89]:


dtrain = xgb.DMatrix(x_train, label = y_t1)
dtest = xgb.DMatrix(x_test, label = y_t2)


# In[90]:


#creating watchlist of training
# to see out output 
watchlist = [(dtrain,'train'),(dtest, 'eval')]


# In[92]:


params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': True,    # would be boolean in sklearn
    'booster' : 'gbtree',
    'max_depth' : 7,
    'eval_metric' : 'auc'
}
# using bydeafault eta [default=0.3, alias: learning_rate]

num_rounds = 100  


# In[94]:


model_xgb = xgb.train(params, dtrain, num_rounds, evals = watchlist, early_stopping_rounds = 15, verbose_eval = True)


# The train-auc:0.94706   and Test-auc:0.917142

# In[100]:


features_contribution = pd.Series(model_xgb.get_fscore()).sort_values(ascending=False)


# In[107]:


plt.figure(figsize=(20,10))
features_contribution[:50].plot(kind='bar', title='Feature Importances')
#features_contribution[:50].plot(kind='line')
plt.ylabel('Feature Importance Score')
plt.show()


# https://xgboost.readthedocs.io/en/latest/parameter.html
