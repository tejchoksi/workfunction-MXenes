import os
os.environ['PYTHONHASHSEED']=str(2)

import numpy as np
import pandas as pd
import tensorflow as tf
import copy

from itertools import combinations
from sklearn import preprocessing
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2)

dataset = pd.read_excel(r"C2DB FINAL DATA.xlsx", engine = 'openpyxl')

f = open('Output.txt','a')

###############

'''
A list with 10 features is created.
Then, 5-feature combinations are created, which are then evaluated on the Neural Network Regression model.
The models are evaluated using MAE, and arranged based on their performance.
The top 25 5-feature models are selected, and the count of each feature is used to find out Occurrence Probability.
'''


y = dataset['Workfunction']

X = dataset.drop(columns = ['Workfunction','Formula','r-(X)','EA(X)','IP(T)','EA(T)','EN(X)'])

Element_df = pd.concat([X.pop(x) for x in ['M', 'X','T']], axis=1)

X_Copy = copy.deepcopy(X)

############### Creating 5-Feature combinations

Feature_list = ['r-(M)','r-(T)','EA(M)','IP(M)','IP(X)','EN(M)','EN(T)','Length(x)','HoF','E_Hull']

reset_random_seeds()

comb = combinations(Feature_list,5)

comb_list = []

for i in comb:
    i = list(i)
    comb_list.append(i)

############### K-Fold

kf = KFold(n_splits=5, random_state=None)

############### Neural Network Regression

reset_random_seeds()

Input_Shape = [5]

MAE_avg = []

F1 = []
F2 = []
F3 = []
F4 = []
F5 = []

for i,x in enumerate(comb_list): 
    
    X = X.drop(columns = x)

    feature_columns = X.columns
    
    F1.append(feature_columns[0])
    F2.append(feature_columns[1])
    F3.append(feature_columns[2])
    F4.append(feature_columns[3])
    F5.append(feature_columns[4])

    X = preprocessing.normalize(X)

    NN_pred = [None]*275
    count = 0
    CV_scores = []
    
    for train_index , test_index in kf.split(X):
    
        X_train , X_test = X[train_index,:],X[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        NN_model = keras.Sequential([
        layers.BatchNormalization(input_shape = Input_Shape),
        layers.Dense(512, activation = 'relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, activation = 'relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation = 'sigmoid'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation = 'sigmoid'),
        layers.Dense(1)
        ])

        NN_model.compile(optimizer = 'adam', loss = 'mae')
        
        NN_model.fit(X_train,y_train, batch_size = 100, epochs = 600, validation_data = (X_test,y_test), verbose = 0)
        pred_values = NN_model.predict(X_test)
    
        for i,v in enumerate(pred_values):
            NN_pred[(i + count*55)] = v
        
        count += 1
    
        scores = NN_model.evaluate(X_test, y_test, verbose=0)
        CV_scores.append(scores)
    
    CV_scores = np.array(CV_scores)
    
    MAE_avg.append(CV_scores.mean())
    
    NN_pred = np.concatenate(NN_pred)
    
    X = copy.deepcopy(X_Copy)

############## Exporting the data

Iterations = list(range(252))
    
DF = {'Iteration':Iterations, 'MAE':MAE_avg, 'Feature 1':F1,'Feature 2':F2,'Feature 3':F3,'Feature 4':F4,'Feature 5':F5}

DF = pd.DataFrame(DF)

DF.sort_values(by = 'MAE', ascending = True, inplace = True)

Final_DF = copy.deepcopy(DF.iloc[:25,:])

print('Top 25 Models -', file = f)
print(Final_DF, file = f)

Feature_Count_List = []

for i in range(5):
    
    Feature_Count_List.append(Final_DF.iloc[:,2+i].tolist())

Feature_Count_df = pd.DataFrame({'Features':(np.concatenate(Feature_Count_List))})

print('Occurrence Probabilities of the Features - ',file = f)
print(Feature_Count_df.Features.value_counts(1), file = f)

