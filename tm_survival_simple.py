import h5py as h5
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
from auton_survival.preprocessing import Scaler
from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc


shots = np.load('/projects/EKOLEMEN/survival_tm/shots.npy')
tm_shots = np.load('/projects/EKOLEMEN/survival_tm/tm_shots.npy')
st_shots = np.load('/projects/EKOLEMEN/survival_tm/st_shots.npy')

def load_data(data_type):
    with open(f'/projects/EKOLEMEN/survival_tm/formatted_labels/{data_type}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return data['x'], data['t'], data['e']


n = len(shots)

tr_size = int(n*0.80)
vl_size = int(n*0.10)
te_size = int(n*0.10)

train_shots = shots[:tr_size]
test_shots = shots[-te_size:]
valid_shots = shots[tr_size:tr_size+vl_size]

x_train, t_train, e_train = load_data('train')
x_test,  t_test,  e_test  = load_data('test')
x_valid, t_valid, e_valid = load_data('valid')

# Get inds for time <600ms
'''inds = np.where(t_train < 600)[0]

x_train = x_train[inds]
t_train = t_train[inds]
e_train = e_train[inds]'''

tm_inds = np.where(e_train == 1)[0]
st_inds = np.where(e_train == 0)[0]

# get same number of tearing mode and stable shots
new_st_inds = np.random.choice(st_inds, size=len(tm_inds), replace=False)

x_train = np.concatenate((x_train[tm_inds], x_train[new_st_inds]), axis=0)
t_train = np.concatenate((t_train[tm_inds], t_train[new_st_inds]), axis=0)
e_train = np.concatenate((e_train[tm_inds], e_train[new_st_inds]), axis=0)

# Shuffle arrays because currently all 1s followed by all 0s
p = np.random.permutation(len(t_train))
x_train = x_train[p,:]
t_train = t_train[p]
e_train = e_train[p]

x_train_df = pd.DataFrame(x_train)
t_train_df = pd.DataFrame(t_train)
e_train_df = pd.DataFrame(e_train)
outcomes_df = pd.DataFrame({'time': t_train, 'event': e_train})
x_valid_df = pd.DataFrame(x_valid)
t_valid_df = pd.DataFrame(t_valid)
e_valid_df = pd.DataFrame(e_valid)

x_test_df = pd.DataFrame(x_test)
t_test_df = pd.DataFrame(t_test)
e_test_df = pd.DataFrame(e_test)

outcomes_valid_df = pd.DataFrame({'time': t_valid, 'event': e_valid})

# normalize

scaler = Scaler()
transformer = scaler.fit(x_train_df)
x_train_df = transformer.transform(x_train_df)
x_test_df = transformer.transform(pd.DataFrame(x_test))

param_grid = {'k' : [2, 3, 4, 6, 8],
              'iters': [1],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [ 1e-5, 1e-4, 1e-3 ],
              'batch_size' : [100, 1000, 10000],
              'layers' : [[50, 100, 150], 
                          [30, 40, 50, 60], 
                          [100], 
                          [100, 100], 
                          [100, 60, 175, 225, 120]],
             }

'''param_grid = {'k' : [3],
              'iters': [3],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-3],
              'batch_size' : [100, 10000],
              'layers' : [[100, 60, 175, 225, 120]]
             }'''

params = ParameterGrid(param_grid)
times = [20, 50, 100, 200]
models=[]
for param in params:
    print(param)
    model = SurvivalModel(model='dsm', 
                      iters=param['iters'], 
                      k=param['k'], 
                      layers=param['layers'], 
                      distribution=param['distribution'],
                      learning_rate=param['learning_rate'], 
                      batch_size=param['batch_size']
                    )
    '''model = SurvivalModel(model='dsm', 
                          iters=param['iters'], 
                          k=param['k'], 
                          layers=param['layers'], 
                          distribution=param['distribution'],
                          learning_rate=param['learning_rate'], 
                          batch_size=param['batch_size']
                        )'''
    _, train_loss, val_loss = model.fit(x_train_df, outcomes_df, val_data=(x_valid_df, outcomes_valid_df))

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
    predictions_val = model.predict_survival(x_valid_df, times)
    metric_val = survival_regression_metric('ibs', outcomes_valid_df, predictions_val, times)
    models.append([metric_val, train_loss, val_loss, model])

    with open('models.pkl', 'wb') as f:
        pickle.dump(models, f)
