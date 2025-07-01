import torch
import gc
import h5py as h5
import matplotlib.pyplot as plt
import os
import configparser
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
import optuna
from sklearn.model_selection import ParameterGrid
#sys.path.append('/projects/EKOLEMEN/survival_tm/train_models/auton-survival')
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
import os
auton_path = os.path.expanduser("~/TMPredictor/survival_tm/auton-survival")
os.environ["PYTHONPATH"] = f"{auton_path}:" + os.environ.get("PYTHONPATH", "")

from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from get_survival_from_shot import get_rt_survival_from_shot, get_cakenn_survival_from_shot
import metrics_helpers
import plotting_helpers
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mpmath.functions.zetazeros import search_supergood_block

import metrics_helpers
from functools import partial
import ast
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.progress_reporter import CLIReporter
from ray.train import report

def validate(model, x, t, e, shots_list):
    time_horizon = 500
    survival_output = model.predict_survival(pd.DataFrame(x), [time_horizon])
    thresholds = np.linspace(0, 1, 100)
    fprs = []
    fnrs = []
    warning_times_list = []
    jumps_list = []
    classification_list = []
    for threshold in thresholds:
        classification, warning_times, jumps, shots, fpr, fnr = metrics_helpers.classify_database(metrics_helpers.LP_filter(1-survival_output, alpha=0.8), e, t, shots_list, threshold)
        fprs.append(fpr)
        fnrs.append(fnr)
        warning_times_list.append(warning_times)
        jumps_list.append(jumps)
        classification_list.append(classification)

    fprs = np.array(fprs)
    fnrs = np.array(fnrs)

    new_tprs = []
    new_fprs = []
    warning_times = []
    for i in range(len(thresholds)):
        warning_times_nonan = [wt for wt in warning_times_list[i] if wt is not None]
        jumps = jumps_list[i]
        new_classification_list = classification_list[i].copy()
        for j in range(len(classification_list[i])):
            if classification_list[i][j] == -1 and jumps[j] != 0:
                new_classification_list[j] = -2
            elif classification_list[i][j] == 0 and jumps[j] != 0:
                new_classification_list[j] = 1
        tps = np.sum(np.array(new_classification_list)==1) - np.sum(np.array(warning_times_nonan)<100)
        fns = np.sum(np.array(new_classification_list)==0)
        tns = np.sum(np.array(new_classification_list)==-1)
        fps = np.sum(np.array(new_classification_list)==-2)

        tpr = tps/(tps+fns)
        fpr = fps/(fps+tns)
        new_tprs.append(tpr)
        new_fprs.append(fpr)
        warning_times.append(np.median(warning_times_nonan))

    return metrics_helpers.get_auc(new_fprs, new_tprs), np.median(warning_times)


def objective(config_filename, model_params):
    # Data Setup
    # LOAD DATABASE

    x_dir = config['model']['database_x_name']
    e_dir = config['model']['database_e_name']
    t_dir = config['model']['database_t_name']
    shots_list_dir = config['model']['database_shots_list_name']
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

    # select random 80% of shots for training, 10% for validation, and 10% for testing
    rng = np.random.default_rng(model_params['seed'])  # Set a random seed for reproducibility
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

    # Model Setup
    model = SurvivalModel(
        model='dsm', iters=model_params['iters'],
        k=model_params['k'],
        layers=[model_params['neurons'], ] * model_params['layers'],
        distribution=model_params['distribution'],
        learning_rate=model_params['learning_rate'],
        batch_size=model_params['batch_size'],
        dropout=model_params['dropout'],
    )
    _, train_loss, val_loss = model.fit(x_train_df, outcomes_train_df, val_data=(x_valid_df, outcomes_valid_df))

    train_score, train_warning_times = validate(model, x_train_df, t_train, e_train,
                           shots_list_train)
    val_score, val_warning_times = validate(model, x_valid_df, t_valid, e_valid, shots_list_valid)
    

    # Explicitly delete big objects
    del model, x_train_df, x_valid_df, outcomes_train_df, outcomes_valid_df

    # Clean up any cyclic references
    gc.collect()
    # Send the current training result back to Tune
    print({"train_loss": train_loss[-1], "val_loss": val_loss[-1],
           "train_score": train_score, "val_score": val_score})
    report({"train_loss": train_loss[-1], "val_loss": val_loss[-1],
                 "train_score": train_score, "val_score": val_score, "val_median_warning_time": val_warning_times})

if __name__ == "__main__":
    if (len(sys.argv)-1) > 0:
        config_filename=sys.argv[1]
    else:
        config_filename='hyperparam_model.cfg'
    # Config parser
    config = configparser.ConfigParser()
    config.read(config_filename)
    search_space = {}
    for param in config['training'].keys():
        search_space[param] = eval(config['training'][param])


    # Define the scheduler and reporter
    scheduler = ASHAScheduler(metric="val_score", mode="max", max_t=500,
                              grace_period=25, reduction_factor=2)

    reporter = CLIReporter(metric_columns=["train_loss", "val_loss",
                                           "train_score", "val_score"],
                           max_progress_rows=100)

    new_objective = partial(objective, config_filename)

    # Run the tuning
    result = tune.run(
        new_objective, resources_per_trial={"cpu": 10}, max_concurrent_trials=5,
        name=config['model']['output_filename_base'], 
        trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        config=search_space, num_samples=1000, scheduler=scheduler,
        progress_reporter=reporter, resume="AUTO+ERRORED")
