import h5py as h5
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pickle
import os

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

data = h5.File('/projects/EKOLEMEN/profile_predictor/joe_hiro_models/diiid_data.h5', 'r')
tm_data = h5.File('/projects/EKOLEMEN/survival_tm/tm_labels.h5', 'r')
shots = np.load('/projects/EKOLEMEN/survival_tm/tm_shots.npy').astype(str)

prof_signals = ['zipfit_etempfit_rho', 'zipfit_itempfit_rho', 'zipfit_edensfit_rho', 'zipfit_trotfit_rho', 'qpsi_EFIT01']

signals = ['pinj', 'tinj', 'betan_EFIT01', 'qmin_EFIT01', 'ech_pwr_total', 'ip', 'bt', 'li_EFIT01', 'aminor_EFIT01', 
           'rmaxis_EFIT01', 'tribot_EFIT01', 'tritop_EFIT01', 'kappa_EFIT01', 'volume_EFIT01']

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

def gather_feature(shot_data, prof_signals, signals, ind):
    feature = np.zeros(len(prof_signals)*33+len(signals))
    
    # Load scalar signals
    for i, sig in enumerate(signals):
        feature[i] = shot_data[sig][ind]
    
    # Load profile signals
    for i, sig in enumerate(prof_signals):
        feature[33*i+len(signals):33*(i+1)+len(signals)] = shot_data[sig][ind,:]
    
    return feature

if __name__ == '__main__':

    for shot in shots[10000*idx:10000*(idx+1)]:
        if int(shot) % 100 == 0:
            print(shot)
        
        x = [] # Feature vector
        t = [] # Time to measurment
        e = [] # Event of measurement (0 if no TM/end of shot, 1 if TM occurred)
        
        try:
            # First figure out time of TM, or time of end of shot
            label = tm_data[shot]['label'][:]
            time = tm_data[shot]['time'][:]
            tm_ind = np.argmax(label>0)
            tm_time = time[tm_ind]

            # Find start and stop of flattop
            start = np.array(data[shot]['t_ip_flat_sql'])
            duration = np.array(data[shot]['ip_flat_duration_sql'])
            
            # If either is nan value, continue
            if np.isnan(start) or np.isnan(duration):
                continue
            
            stop = start+duration
            # Get start and stop inds
            start_ind = np.argmin(np.abs(time - start))
            end_ind = np.argmin(np.abs(time - stop))

            # End shot when TM occurs
            if end_ind > tm_ind and tm_ind != 0:
                end_ind = tm_ind
            
            # Go through valid indices and gather dataset
            for ind in range(start_ind, end_ind):
                # Check that we have all signals at this time
                if check_all_signals(data[shot], prof_signals, signals, ind):
                    feature = gather_feature(data[shot], prof_signals, signals, ind)
                    x.append(feature)

                    if tm_time > 0:
                        # Time to TM
                        t.append(tm_time - time[ind])
                        e.append(1)
                    else:
                        # Time to end of shot
                        t.append(time[end_ind] - time[ind])
                        e.append(0)

            output = dict()
            output['x'] = x
            output['t'] = t
            output['e'] = e
            output['shot'] = shot

            with open(f'/projects/EKOLEMEN/survival_tm/survival_labels/{shot}.pkl', 'wb') as f:
                pickle.dump(output, f)
        except: 
            print(f'Error on shot {shot}')