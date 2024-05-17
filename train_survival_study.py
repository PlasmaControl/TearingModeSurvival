import h5py as h5
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pickle
import sys
import optuna

sys.path.append('/home/ar0535/auton-survival/')

from sklearn.model_selection import ParameterGrid
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

shots = np.load('/projects/EKOLEMEN/survival_tm/shots.npy')

def load_data(data_type):
    with open(f'/projects/EKOLEMEN/survival_tm/formatted_labels/{data_type}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return data['x'], data['t'], data['e']

def objective(trial):
    activation = trial.suggest_categorical('activation', ['ReLU6', 'ReLU', 'SeLU', 'Tanh'])
    # dist = trial.suggest_categorical('distribution', ['LogNormal', 'Weibull'])
    dist = 'LogNormal'
    k = trial.suggest_int('k', 2, 10)
    lr = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
    num_layers = trial.suggest_int('layers', 1, 5)
    bs = trial.suggest_int('batch', 10000, 40000)
    
    l1 = trial.suggest_int('lin1', 25, 250)
    l2 = trial.suggest_int('lin2', 25, 250)
    l3 = trial.suggest_int('lin3', 25, 250)
    l4 = trial.suggest_int('lin4', 25, 250)
    l5 = trial.suggest_int('lin5', 25, 250)
    
    if num_layers == 1:
        layers = [l1]
    elif num_layers == 2:
        layers = [l1 ,l2]
    elif num_layers == 3:
        layers = [l1 ,l2, l3]
    elif num_layers == 4:
        layers = [l1 ,l2, l3, l4]
    elif num_layers == 5:
        layers = [l1 ,l2, l3, l4, l5]
    
    model = DeepSurvivalMachines(k = k,
                                 distribution = dist,
                                 layers = layers, 
                                 activation = activation)
    # The fit method is called to train the model
    _, train_loss, valid_loss = model.fit(x_train, t_train, e_train, iters = 50, val_data=(x_valid, t_valid, e_valid),
              batch_size=bs, learning_rate = lr)
    
    err = model.compute_nll(x_valid, t_valid, e_valid)
    return err

if __name__ == '__main__':
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
    """
    inds = np.where(t_train < 600)[0]

    x_train = x_train[inds]
    t_train = t_train[inds]
    e_train = e_train[inds]
    """

    tm_inds = np.where(e_train == 1)[0]
    st_inds = np.where(e_train == 0)[0]
    new_st_inds = np.random.choice(st_inds, size=len(tm_inds), replace=False)
    
    x_train = np.concatenate((x_train[tm_inds], x_train[new_st_inds]), axis=0)
    t_train = np.concatenate((t_train[tm_inds], t_train[new_st_inds]), axis=0)
    e_train = np.concatenate((e_train[tm_inds], e_train[new_st_inds]), axis=0)
    
    # Shuffle arrays because currently all 1s followed by all 0s
    p = np.random.permutation(len(t_train))
    x_train = x_train[p,:]
    t_train = t_train[p]
    e_train = e_train[p]
    
    study = optuna.create_study()
    for num_trials in range(500):
        study.optimize(objective, n_trials=20)
        
        # Save study
        with open('/home/ar0535/tm_study.pkl', 'wb') as f:
            pickle.dump(study, f)