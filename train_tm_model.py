import configparser
import os
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel
import sys
import ast

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='model.cfg'

config = configparser.ConfigParser()
config.read(config_filename)

dumping_dir = config['model']['output_filename_base']
x_dir = config['model']['database_x_name']
e_dir = config['model']['database_e_name']
t_dir = config['model']['database_t_name']
shots_list_dir = config['model']['database_shots_list_name']

print(f'x_dir: {x_dir}')
print(f'e_dir: {e_dir}')
print(f't_dir: {t_dir}')
with open(f'/projects/EKOLEMEN/survival_tm_2/data/{x_dir}.pkl', 'rb') as f:
    x = np.array(pickle.load(f))
with open(f'/projects/EKOLEMEN/survival_tm_2/data/{e_dir}.pkl', 'rb') as f:
    e = np.array(pickle.load(f))
with open(f'/projects/EKOLEMEN/survival_tm_2/data/{t_dir}.pkl', 'rb') as f:
    t = np.array(pickle.load(f))
with open(f'/projects/EKOLEMEN/survival_tm_2/data/{shots_list_dir}.pkl', 'rb') as f:
    shots_list = np.array(pickle.load(f))
# 80, 10, 10 train, valid, test split
# x_train_df = pd.DataFrame(x[:int(len(x)*0.8)])
# x_valid_df = pd.DataFrame(x[int(len(x)*0.8):int(len(x)*0.9)]).reset_index(drop=True)
# x_test_df = pd.DataFrame(x[int(len(x)*0.9):]).reset_index(drop=True)
# shots_list_train = shots_list[:int(len(x)*0.8)]
# shots_list_valid = shots_list[int(len(x)*0.8):int(len(x)*0.9)]
# shots_list_test = shots_list[int(len(x)*0.9):]
# e_train = e[:int(len(x)*0.8)]
# e_valid = e[int(len(x)*0.8):int(len(x)*0.9)]
# e_test = e[int(len(x)*0.9):]
# t_train = t[:int(len(x)*0.8)]
# t_valid = t[int(len(x)*0.8):int(len(x)*0.9)]
# t_test = t[int(len(x)*0.9):]

# outcomes_train_df = pd.DataFrame({'time': t_train, 'event': e_train})
# outcomes_valid_df = pd.DataFrame({'time': t_valid, 'event': e_valid})
    
unique_shots = np.unique(shots_list)
# find the index of the first occurrence of each unique shot
shots_indices = [np.where(shots_list == shot)[0][0] for shot in unique_shots]
shots_to_index_dict = {}
for i, shot in enumerate(unique_shots):
    shots_to_index_dict[shot] = np.arange(shots_indices[i], shots_indices[i+1]) if i < len(shots_indices) - 1 else np.arange(shots_indices[i], len(shots_list))

param_grid = {
    'iters': ast.literal_eval(config['training']['iters']),
    'k': ast.literal_eval(config['training']['k']),
    'layers': ast.literal_eval(config['training']['layers']),
    'distribution': ast.literal_eval(config['training']['distribution']),
    'learning_rate': ast.literal_eval(config['training']['learning_rate']),
    'batch_size': ast.literal_eval(config['training']['batch_size']),
    'dropout': ast.literal_eval(config['training']['dropout']),
    'seed': ast.literal_eval(config['training']['seed'])
             }
params = ParameterGrid(param_grid)
# select random 80% of shots for training, 10% for validation, and 10% for testing

rng = np.random.default_rng(params[0]['seed'])  # Set a random seed for reproducibility
shots_train = rng.choice(unique_shots, size=int(len(unique_shots)*0.8), replace=False)
shots_valid = rng.choice(np.setdiff1d(unique_shots, shots_train), size=int(len(unique_shots)*0.1), replace=False)
shots_test = np.setdiff1d(unique_shots, np.concatenate((shots_train, shots_valid)))

indices_train = np.concatenate([shots_to_index_dict[shot] for shot in shots_train])
indices_valid = np.concatenate([shots_to_index_dict[shot] for shot in shots_valid])
indices_test = np.concatenate([shots_to_index_dict[shot] for shot in shots_test])

x_train_df = pd.DataFrame(x[indices_train]).reset_index(drop=True)
x_valid_df = pd.DataFrame(x[indices_valid]).reset_index(drop=True)
x_test_df = pd.DataFrame(x[indices_test]).reset_index(drop=True)
shots_list_train = shots_list[indices_train]
shots_list_valid = shots_list[indices_valid]
shots_list_test = shots_list[indices_test]
e_train = e[indices_train]
e_valid = e[indices_valid]
e_test = e[indices_test]
t_train = t[indices_train]
t_valid = t[indices_valid]
t_test = t[indices_test]
outcomes_train_df = pd.DataFrame({'time': t_train, 'event': e_train})
outcomes_valid_df = pd.DataFrame({'time': t_valid, 'event': e_valid})

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
                      batch_size=param['batch_size'],
                      dropout=param['dropout'],
                    )
    _, train_loss, val_loss = model.fit(x_train_df, outcomes_train_df, val_data=(x_valid_df, outcomes_valid_df))
    models.append([model, train_loss, val_loss, param])
    print(f'dumping to models/{dumping_dir}.pkl')
    with open(f'models/{dumping_dir}.pkl', 'wb') as f:
        pickle.dump(models, f)
