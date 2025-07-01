import numpy as np

def find_peaks_in_data(data):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i-1] < data[i] > data[i+1]:
            peaks.append(i)
    return peaks

def LP_filter(x, alpha=0.8):
    y = np.zeros(x.shape)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = alpha * y[i-1] + (1-alpha) * x[i]
    return y

def get_auc(fprs, tprs):
    pairs = sorted(zip(fprs, tprs), key=lambda x: x[0])
    fprs_sorted = np.array([p[0] for p in pairs])
    tprs_sorted = np.array([p[1] for p in pairs])
    return np.trapz(tprs_sorted, fprs_sorted)

# given the tm probability, event, time and threshold, return TP, FP, TN, FN for the final event and its warning time, 
# and count the number of 'jumps', going from positive to negative prediction
def get_classification(tm_probability, e, t, threshold):
    # label_timetrace is 1 if probability is above threshold and 0 below
    label_timetrace = np.where(tm_probability[:, 0] > threshold, 1, 0)

    # find number of times that label_timetrace goes from 0 to 1 to 0 and lasts 100 indices
    jumps = 0
    for i in range(1, len(label_timetrace)): 
        if label_timetrace[i-1] == 0 and label_timetrace[i] == 1:
            # find the next index that goes to 0
            next_zero_index = np.where(label_timetrace[i:] == 0)[0]
            if len(next_zero_index) > 0 and np.abs(t[i] - t[next_zero_index[0]]) >= 400:
                # if the next zero index is at least 400ms away, count it as a jump
                jumps += 1

    # to find classification, take last label time and compare to event
    # 1 means correct TM prediction (TP), 0 means unpredicted TM (FN), -1 means no TM in shot and no TM predicted (TN), -2 means no TM in shot and yes TM predicted (FP)
    if label_timetrace[-1] == 1 and e[-1] == 1:
        classification = 1
    elif label_timetrace[-1] == 1 and e[-1] == 0:
        classification = -2
    elif label_timetrace[-1] == 0 and e[-1] == 1:
        classification = 0
    elif label_timetrace[-1] == 0 and e[-1] == 0:
        classification = -1

    # find warning time
    warning_time = None # only at true positive is there a warning time
    if classification == 1:
        # if tm is always predicted, warning time is length of shot
        if np.all(label_timetrace == 1):
            warning_time = t[0]
        else:
            # find index at which labels switch to 1
            warning_index = len(label_timetrace) - np.where(label_timetrace[::-1] == 0)[0][0]
            warning_time = t[warning_index]

    return classification, warning_time, jumps

def classify_database(tm_probability, e, t, shots_list, threshold):
    shot_indices = find_peaks_in_data(t)
    classifications = []
    warning_times = []
    jumps = []
    shots = []
    for i in range(len(shot_indices) - 1):
        shot_start = shot_indices[i]
        shot_end = shot_indices[i+1]
        shot = shots_list[i]
        shot_tm_probability = tm_probability[shot_start:shot_end]
        shot_e = e[shot_start:shot_end]
        shot_t = t[shot_start:shot_end]
        classification, warning_time, jump = get_classification(shot_tm_probability, shot_e, shot_t, threshold)
        classifications.append(classification)
        warning_times.append(warning_time)
        jumps.append(jump)
        shots.append(shot)
    fpr = classifications.count(-2) / (classifications.count(-2) + classifications.count(-1))
    fnr = classifications.count(0) / (classifications.count(0) + classifications.count(1))
    return classifications, warning_times, jumps, shots, fpr, fnr

def fnr_fpr_calculator(model, normed_x, normed_t, normed_e, prediction_times, threshold=0.7):
    out_survival = model.predict_survival(normed_x, prediction_times)
    fnrs = []
    fprs = []
    shot_indices = find_peaks_in_data(normed_t)
    for i, time in enumerate(prediction_times):
        tm_prediction_per_shot = []
        # 1 means correct TM prediction (TP), 0 means unpredicted TM (FN), -1 means no TM in shot and no TM predicted (TN), -2 means no TM in shot and yes TM predicted (FP)
        # a TM is predicted when the survival prediction is 0 at any point in the shot. Check if better results when TM is consecutive 0s
        survival_prediction = out_survival[:,i]
        survival_prediction = (survival_prediction > threshold).astype(int)
        for j, shot_index in enumerate(shot_indices[:-1]):
            tm = (0 in survival_prediction[shot_indices[j]:shot_indices[j+1]])
            if normed_e[shot_index] == 1 and tm:
                tm_prediction_per_shot.append(1)
            elif normed_e[shot_index] == 1 and not tm:
                tm_prediction_per_shot.append(0)
            elif normed_e[shot_index] == 0 and not tm:
                tm_prediction_per_shot.append(-1)
            elif normed_e[shot_index] == 0 and tm:
                tm_prediction_per_shot.append(-2)
        fnr = tm_prediction_per_shot.count(0) / (tm_prediction_per_shot.count(1) + tm_prediction_per_shot.count(0))
        fpr = tm_prediction_per_shot.count(-2) / (tm_prediction_per_shot.count(-1) + tm_prediction_per_shot.count(-2))
        #print(f'fnr: {fnr} from {tm_prediction_per_shot.count(1) + tm_prediction_per_shot.count(0)} shots, fpr: {fpr} from {tm_prediction_per_shot.count(-1) + tm_prediction_per_shot.count(-2)} shots')
        fnrs.append(fnr)
        fprs.append(fpr)
    auc_fnr = np.trapz(fnrs, prediction_times)
    auc_fpr = np.trapz(fprs, prediction_times)
    return auc_fpr, auc_fnr, fprs, fnrs, prediction_times