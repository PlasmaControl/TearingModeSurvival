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
import optuna
from sklearn.model_selection import ParameterGrid
#sys.path.append('/projects/EKOLEMEN/survival_tm/train_models/auton-survival')
sys.path.append(os.path.expanduser("~/TMPredictor/survival_tm/auton-survival"))
from sklearn.model_selection import ParameterGrid
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from sklearn.decomposition import PCA

def check_all_signals(shot_data, prof_signals, signals, ind):
    # Check scalars for nan values
    for sig in signals:
        if np.isnan(shot_data[sig][ind]):
            return False
    
    # Check profiles for nan values
    for sig in prof_signals:
        if np.sum(np.isnan(shot_data[sig][ind,:])) > 0:
            return False
    
    return True

def gather_feature_cakenn(shot_data, prof_signals, signals, ind):
    feature = np.zeros(len(prof_signals)*121+len(signals))
    
    # Load scalar signals
    for i, sig in enumerate(signals):
        feature[i] = shot_data[sig][ind]
    
    # Load profile signals
    for i, sig in enumerate(prof_signals):
        feature[121*i+len(signals):121*(i+1)+len(signals)] = shot_data[sig][ind,:]
    
    return feature

def gather_feature_rt(shot_data, prof_signals, signals, ind):
    feature = np.zeros(len(prof_signals)*33+len(signals))
    
    # Load scalar signals
    for i, sig in enumerate(signals):
        feature[i] = shot_data[sig][ind]
    
    # Load profile signals
    for i, sig in enumerate(prof_signals):
        feature[33*i+len(signals):33*(i+1)+len(signals)] = shot_data[sig][ind,:]
    
    return feature

def get_rt_survival_from_shot(shot, data_filename, normalizations_dict, model_name, time_horizon):
        
    with open(f'data/{data_filename}.pkl', 'rb') as f:
        data = pickle.load(f)
    # faulty shot
    if '199508' in data.keys():
        data['199508']['ech_pwr_total'] = data['199508'].pop('TPSECH')
    prof_signals = ['thomson_temp_mtanh_1d', 'cer_temp_csaps_1d', 'thomson_density_mtanh_1d', 'cer_rot_csaps_1d', 'qpsi_EFITRT2', 'pres_EFITRT2']

    signals = ['bmspinj', 'bmstinj', 'betan_EFITRT2', 'qmin_EFITRT2', 'ech_pwr_total', 'ip', 'PCBCOIL', 'li_EFITRT2', 'aminor_EFITRT2', 
            'rmaxis_EFITRT2', 'tribot_EFITRT2', 'tritop_EFITRT2', 'kappa_EFITRT2', 'volume_EFITRT2']

    x = []

    times = []

    # Get start and stop inds
    start_ind = 0
    end_ind = 300
    # Go through valid indices and gather dataset
    for ind in range(start_ind, end_ind):
        # Check that we have all signals at this time
        if check_all_signals(data[shot], prof_signals, signals, ind):
            feature = gather_feature_rt(data[shot], prof_signals, signals, ind)
            times.append(ind*20)
            x.append(feature)
            
    ### fix BT
    x_train_df = np.array(x)
    x_train_df[:, 6] = 1.69861e-5 * x_train_df[:, 6]

    ##### normalize
    # 'data/rt_normalizations_dict.pkl'
    with open(f'data/{normalizations_dict}.pkl', 'rb') as f:
        normalizations = pickle.load(f)

    
    # separate each normalized profile
    scalars = x_train_df[:, 0:14]
    te = x_train_df[:, 14:14+33]
    ti = x_train_df[:, 14+33:14+33*2]
    ne = x_train_df[:, 14+33*2:14+33*3]
    rot = x_train_df[:, 14+33*3:14+33*4]
    qinv = x_train_df[:, 14+33*4:14+33*5]
    qinv = 1/qinv
    qinv = np.where(qinv == np.inf, 1, qinv)
    pres = x_train_df[:, 14+33*5:14+33*6]

    x_old = np.linspace(0, 1, 33)
    x_long = np.linspace(0, 1, 100)
    x_for_q = np.linspace(0, 1, 65)
    te = np.array([np.interp(x_long, x_old, te[i]) for i in range(len(te))])
    ti = np.array([np.interp(x_long, x_old, ti[i]) for i in range(len(ti))])
    ne = np.array([np.interp(x_long, x_old, ne[i]) for i in range(len(ne))])
    rot = np.array([np.interp(x_long, x_old, rot[i]) for i in range(len(rot))])
    qinv = np.array([np.interp(x_for_q, x_old, qinv[i]) for i in range(len(qinv))])
    pres = np.array([np.interp(x_for_q, x_old, pres[i]) for i in range(len(pres))])

    # Fit PCA on the data

    pca = PCA(n_components=4)
    pca.components_ = normalizations['thomson_temp_mtanh_1d']['pca_matrix']
    pca.mean_ = normalizations['thomson_temp_mtanh_1d']['pca_mean']
    te_pca = pca.transform(te)

    pca = PCA(n_components=4)
    pca.components_ = normalizations['cer_temp_csaps_1d']['pca_matrix']
    pca.mean_ = normalizations['cer_temp_csaps_1d']['pca_mean']
    ti_pca = pca.transform(ti)


    pca = PCA(n_components=4)
    pca.components_ = normalizations['thomson_density_mtanh_1d']['pca_matrix']
    pca.mean_ = normalizations['thomson_density_mtanh_1d']['pca_mean']
    ne_pca = pca.transform(ne)


    pca = PCA(n_components=4)
    pca.components_ = normalizations['rotation_kms']['pca_matrix']
    pca.mean_ = normalizations['rotation_kms']['pca_mean']
    rot_pca = pca.transform(rot)

    pca = PCA(n_components=4)
    pca.components_ = normalizations['qpsi_EFITRT2']['pca_matrix']
    pca.mean_ = normalizations['qpsi_EFITRT2']['pca_mean']
    qinv_pca = pca.transform(qinv)

    pca = PCA(n_components=4)
    pca.components_ = normalizations['pres_EFITRT2']['pca_matrix']
    pca.mean_ = normalizations['pres_EFITRT2']['pca_mean']
    pres_pca = pca.transform(pres)

    # normalize PCA components

    scalars_mean = []
    scalars_std = []

    for scalar in signals:
        scalars_mean.append(normalizations[scalar]['mean'])
        scalars_std.append(normalizations[scalar]['std'])

    scalars_normed = (scalars - scalars_mean)/scalars_std

    te_pca_mean = normalizations['thomson_temp_mtanh_1d']['mean']
    te_pca_std = normalizations['thomson_temp_mtanh_1d']['std']
    te_pca_normed = (te_pca - te_pca_mean)/te_pca_std

    ti_pca_mean = normalizations['cer_temp_csaps_1d']['mean']
    ti_pca_std = normalizations['cer_temp_csaps_1d']['std']
    ti_pca_normed = (ti_pca - ti_pca_mean)/ti_pca_std

    ne_pca_mean = normalizations['thomson_density_mtanh_1d']['mean']
    ne_pca_std  = normalizations['thomson_density_mtanh_1d']['std']
    ne_pca_normed = (ne_pca - ne_pca_mean)/ne_pca_std

    rot_pca_mean = normalizations['rotation_kms']['mean']
    rot_pca_std = normalizations['rotation_kms']['std']
    rot_pca_normed = (rot_pca - rot_pca_mean)/rot_pca_std

    qinv_pca_mean = normalizations['qpsi_EFITRT2']['mean']
    qinv_pca_std = normalizations['qpsi_EFITRT2']['std']
    qinv_pca_normed = (qinv_pca - qinv_pca_mean)/qinv_pca_std

    pres_pca_mean = normalizations['pres_EFITRT2']['mean']
    pres_pca_std = normalizations['pres_EFITRT2']['std']
    pres_pca_normed = (pres_pca - pres_pca_mean)/pres_pca_std

    x_pca = pd.DataFrame(np.concatenate((scalars_normed, te_pca_normed, ti_pca_normed, ne_pca_normed, rot_pca_normed, qinv_pca_normed, pres_pca_normed), axis=1))


    with open(f'models/{model_name}.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    rt_out_survival_1000 = loaded_models[0][0].predict_survival(x_pca, [time_horizon])

    return times, rt_out_survival_1000, x_pca


def get_cakenn_survival_from_shot(wanted_shot, cakenn_data_filename, scalars_data_filename, normalizations_dict, model_name, time_horizon):
    # load cakenn data 
    with open(f'data/{cakenn_data_filename}.pkl', 'rb') as f:
        cakenn_combined = pickle.load(f)
    true_labels = ['p [kPa]', '1/q', r'j [MA m$^{-2}$]', r'n$_e$ [10$^{19}$ m$^{-3}$]', r'T$_e$ [keV]', r'T$_i$ [keV]', r'V$_{tor}$ [km/s]']
    # this is the cakenn order of signals
    prof_signals = ['p', '1/q', 'j', 'ne', 'Te', 'Ti', 'Vtor']

    # Convert the dictionary to a format that can be used for the model
    def convert_dict(original_dict, profiles):
        new_dict = {}
        times_dict = {}
        for shot, time_dict in original_dict.items():
            # Initialize new dictionary for the current shot
            new_dict[shot] = {profile: [] for profile in profiles}
            times_dict[shot] = list(time_dict.keys())
            for time, data in time_dict.items():
                for i, profile in enumerate(profiles):
                    new_dict[shot][profile].append(data[0][:, i])
            
            # Convert lists to numpy arrays
            for profile in profiles:
                new_dict[shot][profile] = np.array(new_dict[shot][profile])
        
        return new_dict, times_dict

    new_dict, times_dict = convert_dict(cakenn_combined, prof_signals)

    signals = ['bmspinj', 'bmstinj', 'betan_EFITRT2', 'qmin_EFITRT2', 'ech_pwr_total', 'ip', 'PCBCOIL', 'li_EFITRT2', 'aminor_EFITRT2', 
          'rmaxis_EFITRT2', 'tribot_EFITRT2', 'tritop_EFITRT2', 'kappa_EFITRT2', 'volume_EFITRT2']
    # this is the model's order of signals
    prof_signals = ['Te', 'Ti', 'ne', 'Vtor', '1/q', 'p', 'j']

    # load scalars data
    # 'data/recent_data_scalars199596_199610.pkl'
    with open(f'data/{scalars_data_filename}.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    # faulty shot
    if '199508' in data_dict.keys():
        data_dict['199507']['ech_pwr_total'] = data_dict['199507'].pop('TPSECH')
        data_dict['199508']['ech_pwr_total'] = data_dict['199508'].pop('TPSECH')
    
    # combine the scalars with cakenn profiles
    final_dict = {}
    for shot, shot_data in data_dict.items():
        if shot in new_dict.keys():
            final_dict[shot] = {}
            for signal in signals:
                final_dict[shot][signal] = shot_data[signal]
            for profile in new_dict[shot].keys():
                final_dict[shot][profile] = new_dict[shot][profile]
    # add zeros to fill cakenn profiles
    times = np.arange(0, 6000, 20)
    for shot, shot_data in final_dict.items():
        if len(times_dict[shot]) > 0:
            cakenn_time_start = times_dict[shot][0]
            cakenn_time_end = times_dict[shot][-1]
            cakenn_time_start_index = np.where(times == int(cakenn_time_start))[0][0]
            if int(cakenn_time_end) < 6000:
                cakenn_time_end_index = np.where(times == int(cakenn_time_end))[0][0]
                for profile in prof_signals:
                    temp_data = np.zeros((len(times), 121))
                    temp_data[cakenn_time_start_index:cakenn_time_end_index+1, :] = shot_data[profile]
                    shot_data[profile] = temp_data
            else:
                cakenn_time_end_index = np.where(times == 5980)[0][0]
                index_where_cakenn_excedes_6000 = np.where(np.array(times_dict[shot]) == '6000')[0][0]
                for profile in prof_signals:
                    temp_data = np.zeros((len(times), 121))
                    temp_data[cakenn_time_start_index:cakenn_time_end_index+1, :] = shot_data[profile][:index_where_cakenn_excedes_6000, :]
                    shot_data[profile] = temp_data

    x = []

    times = []

    # Get start and stop inds
    start_ind = 0
    end_ind = 300
    # Go through valid indices and gather dataset
    for ind in range(start_ind, end_ind):
        # Check that we have all signals at this time
        if check_all_signals(final_dict[wanted_shot], prof_signals, signals, ind):
            feature = gather_feature_cakenn(final_dict[wanted_shot], prof_signals, signals, ind)
            times.append(ind*20)
            x.append(feature)
            
    ### fix BT
    x_train_df = np.array(x)
    x_train_df[:, 6] = 1.69861e-5 * x_train_df[:, 6]

    with open(f'data/{normalizations_dict}.pkl', 'rb') as f:
        normalizations = pickle.load(f)

    x_pca = np.zeros((len(x_train_df), 14+4*len(prof_signals)))
    print(x_pca.shape)
    for i, scalar_type in enumerate(signals):
        scalars = x_train_df[:, i]

        scalars_mean = normalizations[scalar_type]['mean']
        scalars_std = normalizations[scalar_type]['std']
        scalars_normed = (scalars - scalars_mean)/scalars_std
        x_pca[:, i] = scalars_normed

    for i, profile_type in enumerate(prof_signals):
        profile_data = x_train_df[:, 14 + i*121:14 + (i+1)*121]
        pca = PCA(n_components=4)
        pca.components_ = normalizations[profile_type]['pca_matrix']
        pca.mean_ = normalizations[profile_type]['pca_mean']
        profile_pca = pca.transform(profile_data)

        profile_pca_mean = normalizations[profile_type]['mean'] 
        profile_pca_std = normalizations[profile_type]['std']
        profile_pca_normed = (profile_pca - profile_pca_mean)/profile_pca_std
        x_pca[:, 14 + i*4:14 + (i+1)*4] = profile_pca_normed

    with open(f'models/{model_name}.pkl', 'rb') as file:
        loaded_models = pickle.load(file)

    rt_out_survival_1000 = loaded_models[0][0].predict_survival(pd.DataFrame(x_pca), [time_horizon])

    return times, rt_out_survival_1000, x_pca