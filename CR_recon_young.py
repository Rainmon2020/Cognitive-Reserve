# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:35:07 2025

@author: 20202
"""

import numpy as np
import os
import scipy.io as sio
#from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
from tqdm import tqdm
#import warnings
#from sklearn.exceptions import InconsistentVersionWarning

#warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Hemisphere list
hemi = 'lh'
data_root = '/v16data/user_data/lym/CA_Result/mat/'
new_fmri_root = '/v16data/user_data/lym/CA_Result/new_rfmri/'
results_root = '/v16data/user_data/lym/CA_Result/CR_recon2/'
model_root = '/v16data/user_data/lym/model_result/'

def single_time_pred(fmri_file, save_root, cortex_ind, vertex_values):
    time = int(fmri_file.split('_')[-1].split('.')[0])
    model_save_root = os.path.join(model_root, 'time_' + str(time), 'models/')
    pred_save_root = os.path.join(save_root, 'time_' + str(time), 'pred/')
    os.makedirs(pred_save_root, exist_ok=True)
    #os.makedirs(model_save_root, exist_ok=True)
    #fmri_data = np.load(os.path.join(new_fmri_root, fmri_file)).transpose(1, 0)
    # func = fmri_data[cortex_ind, :].flatten()

    # model = DecisionTreeRegressor()
    # model.fit(basis.flatten().reshape(-1, 1), func)
    # np.save(os.path.join(model_save_root, str(hemi) +  'model.npy'), model)

    for subs in range(243):
        # x_train = np.delete(basis, subs, axis=1)
        # y_train = np.delete(func, subs, axis=1)
        x_test = vertex_values[:, subs]
        model = np.load(os.path.join(model_save_root, str(hemi) +  'model.npy'), allow_pickle=True).item()
        # model.fit(basis.flatten().reshape(-1, 1), func.flatten())
        y_test = model.predict(x_test.reshape(-1, 1))
        np.save(pred_save_root + 'sub-' + str(subs) + '_' + str(hemi) + '_pred.npy', y_test)



    cortex = np.loadtxt(os.path.join(data_root, f'fs_cortex-{hemi}_mask.txt'))
    cortex_ind = np.where(cortex)[0]
    vertex_values = sio.loadmat(os.path.join(data_root, f'sub-hemi-{hemi}_thickness.mat'))['vertex_values']
    basis = vertex_values[cortex_ind, :]

    all_fmri_data = [i for i in os.listdir(new_fmri_root) if hemi in i]

    Parallel(n_jobs=40)(delayed(single_time_pred)(i, results_root, cortex_ind, vertex_values) for i in tqdm(all_fmri_data))
