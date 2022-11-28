import os
os.environ['PYTHONHASHSEED']=str(2)

import numpy as np
import pandas as pd
import gplearn
import copy

from itertools import combinations
from sklearn import preprocessing
from sklearn.model_selection import KFold
from gplearn.genetic import SymbolicTransformer

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   np.random.seed(2)

dataset = pd.read_excel(r"C2DB FINAL DATA.xlsx", engine = 'openpyxl')

f = open('Output2.txt','a')

###############

'''
A list with 8 features is created.
Then, 3-feature combinations are created, which are then evaluated on the Symbolic Transformer model.
Among the combinations, the best performing programs from each generation which have fitness > 0.9 are selected.
These meta-features are arranged in decreasing order of the fitness.
'''


y = dataset['Workfunction']

X = dataset.drop(columns = ['Workfunction','Formula'])

Element_df = pd.concat([X.pop(x) for x in ['M', 'X','T']], axis=1)

X_8F = X.drop(columns = ['E_Hull','HoF','r-(X)','EA(X)','IP(T)','EA(T)','EN(X)'])

X_Copy = copy.deepcopy(X_8F)

############### Creating Feature Combinations

Feature_list = ['r-(M)','r-(T)','EA(M)','IP(M)','IP(X)','EN(M)','EN(T)','Length(x)']

reset_random_seeds()

comb = combinations(Feature_list,5)

comb_list = []

for i in comb:
    i = list(i)
    comb_list.append(i)

############### K-Fold

kf = KFold(n_splits=5, random_state=None)

############### Symbolic Transformer

reset_random_seeds()

Combination_ = []
Fold_No = []
Feature_1 = []
Feature_2 = []
Feature_3 = []
Program_ = []
Fitness_ = []

for i,x in enumerate(comb_list):
    
    print('COMBINATION => ',i,file = f)
    
    X_8F.drop(columns = x)
    
    print(X_8F, file = f)
    
    feature_columns = X_8F.columns
    
    X_8F = preprocessing.normalize(X_8F)    
    
    Fold_Counter = 0
    
    for train_index , test_index in kf.split(X_8F):
    
        X_train , X_test = X_8F[train_index,:],X_8F[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', 'tan']

        gp_t = SymbolicTransformer(generations=20, population_size=1000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=0,
                         random_state=0)

        gp_t.fit(X_train,y_train)
        
        for program in gp_t._best_programs:
            if program.raw_fitness_ > 0.9:
                Combination_.append(i)
                Fold_No.append(Fold_Counter)
                Feature_1.append(feature_columns[0])
                Feature_2.append(feature_columns[1])
                Feature_3.append(feature_columns[2])
                Program_.append(program)
                Fitness_.append(program.raw_fitness_)
            print('The program is - ',program, file = f)
            print(file = f)
            print('Fitness = ',program.raw_fitness_, file = f)
            print(file = f)
        
        Fold_Counter += 1
          
    X_8F = copy.deepcopy(X_Copy)
 
############### Exporting the data
 
Symbolic_Transformer_df = {'Combination':Combination_,'Fold':Fold_No,'Feature 1':Feature_1,'Feature 2':Feature_2,'Feature 3':Feature_3,'Program':Program_,'Fitness':Fitness_}
Symbolic_Transformer_df = pd.DataFrame(Symbolic_Transformer_df)
Symbolic_Transformer_df.sort_values(by = 'Fitness', ascending = False, inplace = True)
Symbolic_Transformer_df.to_excel('Symbolic Transformer Results.xlsx')
