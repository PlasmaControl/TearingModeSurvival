import os
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel

'''with open('data/x_train_memory_normed.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_memory_normed.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_memory.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_memory.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)'''

'''with open('data/x_train_df_normed.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_df_normed.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_df.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_df.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)'''

'''with open('data/x_train_future_normed.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_future_normed.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_future.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_future.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)'''

with open('data/x_train_pca.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_pca.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_df.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_df.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)

param_grid = {'k' : [3],
              'iters': [200],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-5],
              'batch_size' : [1000],
              'layers' : [
                  [1000, 1000]
                  ]
             }

params = ParameterGrid(param_grid)
models=[]
for i, param in enumerate(params):
    print('Hyperparameter ' + str(i) + ' of ' + str(len(params)))
    print(param)
    model = SurvivalModel(model='dsm', 
                      iters=param['iters'], 
                      k=param['k'], 
                      layers=param['layers'], 
                      distribution=param['distribution'],
                      learning_rate=param['learning_rate'], 
                      batch_size=param['batch_size']
                    )
    #_, train_loss, val_loss = model.fit(x_train_df, outcomes_train_df, val_data=(x_valid_df, outcomes_valid_df))
    _, train_loss, val_loss = model.fit(x_train_df, outcomes_train_df)
    models.append([model, train_loss, val_loss, param])

    with open('models/pca_models.pkl', 'wb') as f:
        pickle.dump(models, f)
