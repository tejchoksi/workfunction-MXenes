import os
os.environ['PYTHONHASHSEED']=str(2)

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import copy
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2) 

dataset = pd.read_excel(r"C2DB FINAL DATA.xlsx")

Pure_New_T_Random_Split_dataset = pd.read_excel(r"Pure_New_Terminations_Random_Split.xlsx")

Cyclic_1_dataset = pd.read_excel(r"C2DB FINAL DATA_New_Terminations - Cyclic Split 1.xlsx")

#####################

'''
Feature Selection is done based on the Pearson Correlation Matrix.
We remove the features exhibitng Pearson correlation coefficient less than 0.75, so the number of features is reduced from 15 to 10
The 8 feature model is constructed from the 10 feature model by removing the DFT-derived features (E_Hull and HoF)
'''


y = dataset['Workfunction']

X = dataset.drop(columns = ['Workfunction','Formula','r-(X)','EA(X)','IP(T)','EA(T)','EN(X)'])

Element_df = pd.concat([X.pop(x) for x in ['M', 'X','T']], axis=1)

X_RF_5 = copy.deepcopy(X)

feature_columns = X.columns

X = preprocessing.normalize(X)


Pure_New_T_Random_Split_y = Pure_New_T_Random_Split_dataset['Workfunction']

Pure_New_T_Random_Split_X = Pure_New_T_Random_Split_dataset.drop(columns = ['Workfunction','Formula','r-(X)','EA(X)','IP(T)','EA(T)','EN(X)'])

Pure_New_T_Random_Split_Element_df = pd.concat([Pure_New_T_Random_Split_X.pop(x) for x in ['M', 'X','T']], axis=1)

Pure_New_T_Random_Split_X_RF_5 = copy.deepcopy(Pure_New_T_Random_Split_X)


Cyclic_1_y = Cyclic_1_dataset['Workfunction']

Cyclic_1_X_8F = Cyclic_1_dataset.drop(columns = ['Workfunction','Formula','r-(X)','EA(X)','IP(T)','EA(T)','EN(X)','E_Hull','HoF'])

Cyclic_1_Element_df = pd.concat([Cyclic_1_X_8F.pop(x) for x in ['M', 'X','T']], axis=1)

Cyclic_1_X_8F = preprocessing.normalize(Cyclic_1_X_8F)

###################### K-Fold

kf = KFold(n_splits=5)
    
###################### 10-F Linear Regression


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


###################### 10-F Random Forest Regression


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

################## 10-F Neural Network Regression


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

print('10-F Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

print('Average R2 Score:', Test_r2.mean(),'\n')
print('Train Average R2 Score:', Train_r2.mean(),'\n')

#################### 8-F Neural Network Regression

X_8F = X_RF_5.drop(columns = ['E_Hull','HoF'])
feature_columns_8F = X_8F.columns
X_8F = preprocessing.normalize(X_8F)

Input_Shape = [X_8F.shape[1]]

NN_pred = [None]*275
Train_NN_pred = [None]*1100

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X_8F[train_index,:],X_8F[test_index,:]
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

print('8-F Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

print('Average R2 Score:', Test_r2.mean(),'\n')
print('Train Average R2 Score:', Train_r2.mean(),'\n')

##################### 5-F Neural Network Regression

X_5F = X_RF_5.drop(columns = ['E_Hull','HoF','EN(M)','IP(M)','EA(M)'])
feature_columns_5F = X_5F.columns
X_5F = preprocessing.normalize(X_5F)

Input_Shape = [X_5F.shape[1]]

NN_pred = [None]*275
Train_NN_pred = [None]*1100

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X_5F[train_index,:],X_5F[test_index,:]
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

print('5-F Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

print('Average R2 Score:', Test_r2.mean(),'\n')
print('Train Average R2 Score:', Train_r2.mean(),'\n')

################### Transfer Learning

X_8F = X_RF_5.drop(columns = ['E_Hull','HoF'])
X_8F = preprocessing.normalize(X_8F)

Input_Shape = [X_8F.shape[1]]

count = 0

CV_scores = []
Train_CV_scores = []
Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X_8F):
    
    X_train , X_test = X_8F[train_index,:],X_8F[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    
    reset_random_seeds()
    
    NN_8F_model = keras.Sequential([
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
    layers.Dense(1, name = 'Final')
    ])

    NN_8F_model.compile(optimizer = 'adam', loss = 'mae')

    history = NN_8F_model.fit(X_train,y_train, batch_size = 100, epochs = 600, validation_split = 0.2, verbose = 0)

for layer in NN_8F_model.layers[0:10]:
    layer.trainable = False

for layer in NN_8F_model.layers[10:]:
    layer.trainable = True
        
Pure_New_T_Random_Split_X_8F = Pure_New_T_Random_Split_X_RF_5.drop(columns = ['E_Hull','HoF'])
Pure_New_T_Random_Split_X_8F = preprocessing.normalize(Pure_New_T_Random_Split_X_8F)

Input_Shape = [Pure_New_T_Random_Split_X_8F.shape[1]]

NN_pred = [None]*40


for train_index , test_index in kf.split(Pure_New_T_Random_Split_X_8F):
    
    X_train , X_test = Pure_New_T_Random_Split_X_8F[train_index,:],Pure_New_T_Random_Split_X_8F[test_index,:]
    y_train , y_test = Pure_New_T_Random_Split_y[train_index] , Pure_New_T_Random_Split_y[test_index]
    
    reset_random_seeds()

    NN_8F_model.compile(optimizer = 'adam', loss = 'mae')

    history = NN_8F_model.fit(X_train,y_train, batch_size = 30, epochs = 60, validation_split = 0.2, verbose = 0)
    
    pred_values = NN_8F_model.predict(X_test)
    
    train_pred_values = NN_8F_model.predict(X_train)
    
    for i,v in enumerate(pred_values):
        NN_pred[(i + count*8)] = v
    
    history_DF = pd.DataFrame(history.history)

    history_DF.loc[:, ['loss','val_loss']].plot()
    
    'Mean Absolute Error and R2 score calculation'
    
    scores = NN_8F_model.evaluate(X_test, y_test, verbose=0)
    CV_scores.append(scores)
    
    train_scores = NN_8F_model.evaluate(X_train, y_train, verbose=0)
    Train_CV_scores.append(train_scores)
    
    test_r2_score = r2_score(y_test,NN_pred[(0+8*count):(8 + 8*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_NN_pred[(0+32*count):(32 + 32*count)])
    Train_r2.append(train_r2_score)
    
    count += 1
    
CV_scores = np.array(CV_scores)
Train_CV_scores = np.array(Train_CV_scores)

print('8-F Transfer Learning using Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

################### Augmented Transferability Model

Input_Shape = [Cyclic_1_X_8F.shape[1]]

NN_pred = [None]*315
Train_NN_pred = [None]*1260

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(Cyclic_1_X_8F):
    
    X_train , X_test = Cyclic_1_X_8F[train_index,:],Cyclic_1_X_8F[test_index,:]
    y_train , y_test = Cyclic_1_y[train_index] , Cyclic_1_y[test_index]
    
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

    history = NN_model.fit(X_train,y_train, batch_size = 125, epochs = 500, validation_split = 0.2, verbose = 0)
    pred_values = NN_model.predict(X_test)
    train_pred_values = NN_model.predict(X_train)
    
    for i,v in enumerate(pred_values):
        NN_pred[(i + count*63)] = v
        
    for i,v in enumerate(train_pred_values):
        Train_NN_pred[(i + count*252)] = v
    
    history_DF = pd.DataFrame(history.history)

    history_DF.loc[:, ['loss','val_loss']].plot()
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(NN_pred[(0+63*count):(63 + 63*count)],y_test)
    CV_scores.append(test_score)
    
    train_score = mean_absolute_error(Train_NN_pred[(0+252*count):(252 + 252*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,NN_pred[(0+63*count):(63 + 63*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_NN_pred[(0+252*count):(252 + 252*count)])
    Train_r2.append(train_r2_score)
    
    count += 1

CV_scores = np.array(CV_scores)
Train_CV_scores = np.array(Train_CV_scores)

Test_r2 = np.array(Test_r2)
Train_r2 = np.array(Train_r2)

print('8-F Augmented Neural Network Regression \n')

print('Average MAE Score:', CV_scores.mean(),'\n')
print('Train Average MAE Score:', Train_CV_scores.mean(),'\n')

print('Average R2 Score:', Test_r2.mean(),'\n')
print('Train Average R2 Score:', Train_r2.mean(),'\n')