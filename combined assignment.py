#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import pandas
with open("data/nba_2013.csv", "r") as csvfile:
    nba_raw = pandas.read_csv(csvfile)


# In[ ]:





# In[ ]:


nba = nba_raw.fillna(0)


nba = nba.convert_objects(convert_numeric=True).dropna()
    

print("nba.columns.values:", nba.columns.values)

nba.head(5)


# In[ ]:



import math

selected_player = nba[nba["Player"] == "LeBron James"].iloc[0]
distance_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA',
 '3P.1', '2P', '2PA', '2P.1', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
 'STL', 'BLK', 'TOV', 'PF', 'PTS']

def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)

lebron_distance = nba.apply(euclidean_distance, axis=1)
print("lebron_distance[:5]:\n", lebron_distance[:5])


# In[ ]:


import pandas
with open("data/nba_2013.csv", "r") as csvfile:
    nba_raw = pandas.read_csv(csvfile)



nba_numeric = nba[distance_columns]
nba_numeric.head(5)

nba_normalized = (nba_numeric - nba_numeric.mean()) / nba_numeric.std()
nba_normalized.head(5)


# In[ ]:


from scipy.spatial import distance

# Fill in NA values in nba_normalized.
nba_normalized.fillna(0, inplace=True)

# Find the normalized vector for lebron james.
lebron_normalized = nba_normalized[nba["Player"] == "LeBron James"]

# Find the distance between lebron james and everyone else.
euclidean_distances = nba_normalized.apply(lambda row: distance.euclidean(row, lebron_normalized), axis=1)
distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort("dist", inplace=True)

second_smallest = distance_frame.iloc[1]["idx"]

most_similar_to_lebron = nba.loc[int(second_smallest)]["Player"]
print("most_similar_to_lebron:", most_similar_to_lebron)


# In[ ]:


import random
from numpy.random import permutation

# Randomly shuffle the index of nba.
random_indices = permutation(nba.index)

# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items).
test_cutoff = math.floor(len(nba)/3)

# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = nba.loc[random_indices[1:test_cutoff]]

# Generate the train set with the rest of the data.
train = nba.loc[random_indices[test_cutoff:]]


# In[ ]:



from sklearn.neighbors import KNeighborsRegressor

# The columns that we will be making predictions with.
x_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA',
 '3P.1', '2P', '2PA', '2P.1', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
 'STL', 'BLK', 'TOV', 'PF']
# The column that we want to predict.
y_column = ['PTS']

# Create the knn model.
knn = KNeighborsRegressor(n_neighbors=5)

# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])

# Make predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])

print("predictions[:5]:\n", predictions[:5])


# In[ ]:


actual = test[y_column]

mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print("actual[:20]:\n", actual[:20])
print("mse:", mse)

