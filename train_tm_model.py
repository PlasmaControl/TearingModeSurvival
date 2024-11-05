import os
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel
import sys

dumping_dir = sys.argv[1]
x_dir = sys.argv[2]
e_dir = sys.argv[3]
t_dir = sys.argv[4]
print(f'x_dir: {x_dir}')
print(f'e_dir: {e_dir}')
print(f't_dir: {t_dir}')
with open(f'data/{x_dir}.pkl', 'rb') as f:
    x_train_df = pd.DataFrame(pickle.load(f))
with open(f'data/{e_dir}.pkl', 'rb') as f:
    e = np.array(pickle.load(f))
with open(f'data/{t_dir}.pkl', 'rb') as f:
    t = np.array(pickle.load(f))

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

'''with open('data/x_train_pca.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_pca.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_df.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_df.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)'''

'''with open('data/x_train_pca.pkl', 'rb') as f:
    x_train_df = pickle.load(f)
with open('data/x_valid_pca.pkl', 'rb') as f:
    x_valid_df = pickle.load(f)
with open('data/outcomes_train_manual_norm.pkl', 'rb') as f:
    outcomes_train_df = pickle.load(f)
with open('data/outcomes_valid_manual_norm.pkl', 'rb') as f:
    outcomes_valid_df = pickle.load(f)'''

'''with open('data/rt_x_pca_bms.pkl', 'rb') as f:
    x_train_df = pd.DataFrame(pickle.load(f))
with open('data/rt_filtered_e_bms.pkl', 'rb') as f:
    e = pickle.load(f)
with open('data/rt_filtered_t_bms.pkl', 'rb') as f:
    t = pickle.load(f)'''

'''with open('data/cakenn_x_pca_bms_2.pkl', 'rb') as f:
    x_train_df = pd.DataFrame(pickle.load(f))
with open('data/cakenn_e_bms.pkl', 'rb') as f:
    e = pickle.load(f)
with open('data/cakenn_t_bms.pkl', 'rb') as f:
    t = pickle.load(f)'''

'''with open('data/cakenn_x_pca.pkl', 'rb') as f:
    x_train_df = pd.DataFrame(pickle.load(f))
with open('data/cakenn_e.pkl', 'rb') as f:
    e = pickle.load(f)
with open('data/cakenn_t.pkl', 'rb') as f:
    t = pickle.load(f)'''

outcomes_train_df = pd.DataFrame({'time': t, 'event': e})

param_grid = {'k' : [3],
              'iters': [2500],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-5],
              'batch_size' : [1000],
              'layers' : [
                  [100, 1000]
                  ]
             }

'''param_grid = {'k' : [3, 6, 10],
              'iters': [2500],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-5],
              'batch_size' : [1000],
              'layers' : [
                  [100, 300, 1000], 
                  [300, 300, 300], 
                  [100, 1000], 
                  [50, 80, 100, 150]
                  ]
             }'''

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
    print(f'dumping to models/{dumping_dir}.pkl')
    with open(f'models/{dumping_dir}.pkl', 'wb') as f:
        pickle.dump(models, f)
