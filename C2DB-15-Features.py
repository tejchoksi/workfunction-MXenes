import os
os.environ['PYTHONHASHSEED']=str(2)

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2) 

dataset = pd.read_excel(r"C2DB FINAL DATA.xlsx")


#####################

'''
The complete 15-feature dataset is run on Simple Linear Regression, LASSO, Ridge, Random Forest and Neural Network models.
5-Fold split is used in every single model.
'''


y = dataset['Workfunction']

X = dataset.drop(columns = ['Workfunction','Formula'])

Element_df = pd.concat([X.pop(x) for x in ['M','X','T']], axis=1)

feature_columns = X.columns

X = preprocessing.normalize(X)

###################### K-Fold

kf = KFold(n_splits=5)
    
###################### Simple Linear Regression


LR_model = LinearRegression()

LR_pred = []
Train_LR_pred = []

score_mae = []
Train_score_mae = []

score_r2 = []
Train_score_r2 = []

i = 0
 
for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    LR_model.fit(X_train,y_train)
    pred_values = LR_model.predict(X_test)
    
    Train_pred_values = LR_model.predict(X_train)
    
    for v in pred_values:
        LR_pred.append(v)
        
    for w in Train_pred_values:
        Train_LR_pred.append(w)     
    
    'Mean Absolute Error and R2 score calculation'
    
    score_r2.append(r2_score(y_test,LR_pred[(0+55*i):(55 + 55*i)]))
    Train_score_r2.append(r2_score(y_train,Train_LR_pred[(0+220*i):(220 + 220*i)]))
    
    score_mae.append(mean_absolute_error(LR_pred[(0+55*i):(55 + 55*i)],y_test))
    Train_score_mae.append(mean_absolute_error(Train_LR_pred[(0+220*i):(220 + 220*i)],y_train))
    
    i+=1
    
print('Linear Regression \n')

print('Test Average MAE Score:', sum(score_mae)/len(score_mae),'\n')
print('Train Average MAE Score:', sum(Train_score_mae)/len(Train_score_mae),'\n')

print('Test R2 Score:', sum(score_r2)/len(score_r2),'\n')
print('Train R2 Score:', sum(Train_score_r2)/len(Train_score_r2),'\n')

print('\n')

####################### LASSO Regression


LASSO_model = Lasso()

LASSO_pred = []
Train_LASSO_pred = []

score_mae = []
Train_score_mae = []

score_r2 = []
Train_score_r2 = []

i = 0
 
for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    LASSO_model.fit(X_train,y_train)
    pred_values = LASSO_model.predict(X_test)
    
    Train_pred_values = LASSO_model.predict(X_train)
    
    for v in pred_values:
        LASSO_pred.append(v)
        
    for w in Train_pred_values:
        Train_LASSO_pred.append(w)     
    
    'Mean Absolute Error and R2 score calculation'    
    
    score_mae.append(mean_absolute_error(LASSO_pred[(0+55*i):(55 + 55*i)],y_test))
    Train_score_mae.append(mean_absolute_error(Train_LASSO_pred[(0+220*i):(220 + 220*i)],y_train))
    
    score_r2.append(r2_score(y_test,LASSO_pred[(0+55*i):(55 + 55*i)]))
    Train_score_r2.append(r2_score(y_train,Train_LASSO_pred[(0+220*i):(220 + 220*i)]))
    
    i+=1
    
print('LASSO Regression \n')

print('Test Average MAE Score:', sum(score_mae)/len(score_mae),'\n')
print('Train Average MAE Score:', sum(Train_score_mae)/len(Train_score_mae),'\n')

print('Test R2 Score:', sum(score_r2)/len(score_r2),'\n')
print('Train R2 Score:', sum(Train_score_r2)/len(Train_score_r2),'\n')

print('\n')

##################### Ridge Regression


Ridge_model = Ridge()

Ridge_pred = []
Train_Ridge_pred = []

score_mae = []
Train_score_mae = []

score_r2 = []
Train_score_r2 = []

i = 0
 
for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    Ridge_model.fit(X_train,y_train)
    pred_values = Ridge_model.predict(X_test)
    
    Train_pred_values = Ridge_model.predict(X_train)
    
    for v in pred_values:
        Ridge_pred.append(v)
        
    for w in Train_pred_values:
        Train_Ridge_pred.append(w)     
    
    'Mean Absolute Error and R2 score calculation'    
    
    score_mae.append(mean_absolute_error(Ridge_pred[(0+55*i):(55 + 55*i)],y_test))
    Train_score_mae.append(mean_absolute_error(Train_Ridge_pred[(0+220*i):(220 + 220*i)],y_train))
    
    score_r2.append(r2_score(y_test,Ridge_pred[(0+55*i):(55 + 55*i)]))
    Train_score_r2.append(r2_score(y_train,Train_Ridge_pred[(0+220*i):(220 + 220*i)]))
    
    i+=1

print('Ridge Regression \n')

print('Test Average MAE Score:', sum(score_mae)/len(score_mae),'\n')
print('Train Average MAE Score:', sum(Train_score_mae)/len(Train_score_mae),'\n')

print('Test R2 Score:', sum(score_r2)/len(score_r2),'\n')
print('Train R2 Score:', sum(Train_score_r2)/len(Train_score_r2),'\n')

print('\n')

###################### Random Forest Regression


reset_random_seeds()

RF_model = RandomForestRegressor(criterion='mae')

RF_pred = []
Train_RF_pred = []

score_mae = []
Train_score_mae = []

score_r2 = []
Train_score_r2 = []

i = 0
 
for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    RF_model.fit(X_train,y_train)
    pred_values = RF_model.predict(X_test)
    
    Train_pred_values = RF_model.predict(X_train)
    
    for v in pred_values:
        RF_pred.append(v)
        
    for w in Train_pred_values:
        Train_RF_pred.append(w)     
    
    'Mean Absolute Error and R2 score calculation'    
    
    score_mae.append(mean_absolute_error(RF_pred[(0+55*i):(55 + 55*i)],y_test))
    Train_score_mae.append(mean_absolute_error(Train_RF_pred[(0+220*i):(220 + 220*i)],y_train))
    
    score_r2.append(r2_score(y_test,RF_pred[(0+55*i):(55 + 55*i)]))
    Train_score_r2.append(r2_score(y_train,Train_RF_pred[(0+220*i):(220 + 220*i)]))
    
    i+=1
        
print('Random Forest Regression \n')

print('Test Average MAE Score:', sum(score_mae)/len(score_mae),'\n')
print('Train Average MAE Score:', sum(Train_score_mae)/len(Train_score_mae),'\n')

print('Test R2 Score:', sum(score_r2)/len(score_r2),'\n')
print('Train R2 Score:', sum(Train_score_r2)/len(Train_score_r2),'\n')

print('\n')

################## Neural Network Regression


reset_random_seeds()

Input_Shape = [X.shape[1]]

NN_pred = [None]*275
Train_NN_pred = [None]*1100

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    
    reset_random_seeds()
    
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
    
    history = NN_model.fit(X_train,y_train, batch_size = 100, epochs = 600, validation_split = 0.2, verbose = 0)
    pred_values = NN_model.predict(X_test)
    train_pred_values = NN_model.predict(X_train)
    
    for i,v in enumerate(pred_values):
        NN_pred[(i + count*55)] = v
        
    for i,v in enumerate(train_pred_values):
        Train_NN_pred[(i + count*220)] = v
    
    history_DF = pd.DataFrame(history.history)

    history_DF.loc[:, ['loss','val_loss']].plot()
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(NN_pred[(0+55*count):(55 + 55*count)],y_test)
    CV_scores.append(test_score)
    
    train_score = mean_absolute_error(Train_NN_pred[(0+220*count):(220 + 220*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,NN_pred[(0+55*count):(55 + 55*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_NN_pred[(0+220*count):(220 + 220*count)])
    Train_r2.append(train_r2_score)
    
    count += 1
    
CV_scores = np.array(CV_scores)
Train_CV_scores = np.array(Train_CV_scores)

Test_r2 = np.array(Test_r2)
Train_r2 = np.array(Train_r2)

print('Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

print('Average R2 Score:', Test_r2.mean(),'\n')
print('Train Average R2 Score:', Train_r2.mean(),'\n')