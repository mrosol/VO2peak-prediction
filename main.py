#%%
import pandas as pd
import numpy as np
import logging
import time
from utils import *
# %% Loading data
logging.basicConfig(level=logging.INFO)

id_list = pd.read_csv('ID_in_scope.csv', sep=';')
df = pd.read_csv('treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1 2/test_measure.csv')
info = pd.read_csv('treadmill-maximal-exercise-tests-from-the-exercise-physiology-and-human-performance-lab-of-the-university-of-malaga-1.0.1 2/subject-info.csv')


# %% Selecting relevant data
df = df[df['ID_test'].isin(id_list['0'])]
info = info[info['ID_test'].isin(id_list['0'])]

# %% Calculating features 
time_treadmill_list = []
features_df = {}
hr_threshold = 0.85 #% HRmax
warmup = 30 #s
info_final = []
patients_id = []
for id_test in np.unique(df['ID_test']):
    df_idx = df[df['ID_test']==id_test]
    df_idx = df_idx[df_idx['Speed']>=4.9]
    info_idx = info[info['ID_test']==id_test]
    time_first = df_idx[df_idx['Speed'].diff()>0]['time'].values[0]
    time_last = df_idx[df_idx['Speed'].diff()<0]['time'].values[0]
    time_treadmill = time_last-time_first
    hr_max = max(df_idx['HR'])

    warmup_end = df_idx[df_idx['time']<time_first]['time'].iloc[-1]
    warmup_end_minus_60s = df_idx[df_idx['time']<warmup_end-warmup]['time'].iloc[-1]
    df_idx_run = df_idx[(df_idx['time']>time_first) & (df_idx['time']<time_last)]
    df_idx_run['time'] = df_idx_run['time'] - df_idx_run['time'].iloc[0]
    df_idx_warmup =  df_idx[(df_idx['time']>warmup_end_minus_60s) & (df_idx['time']<warmup_end)]
    df_idx_warmup['time'] = df_idx_warmup['time'] - df_idx_warmup['time'].iloc[0]
    time_treadmill_list.append(time_treadmill)

    df_idx_85_hrmax = select_rows(df_idx_run, hr_threshold, hr_max)
    df_idx_85_hrmax_age = select_rows(df_idx_run, hr_threshold, 220-info_idx['Age'].values[0])

    f_hr_warmup, f_rr_warmup, f_ve_warmup = calcualte_all_features(df_idx_warmup, 'WARMUP')
    f_hr_run_85_hrmax, f_rr_run_85_hrmax, f_ve_run_85_hrmax = calcualte_all_features(df_idx_85_hrmax, 'RUN_85_HRMAX')
    f_hr_run_85_hrmax_age, f_rr_run_85_hrmax_age, f_ve_run_85_hrmax_age = calcualte_all_features(df_idx_85_hrmax_age, 'RUN_85_HRMAX_AGE')

    features_all = pd.concat([
        f_hr_warmup, 
        f_rr_warmup, 
        f_ve_warmup, 
        f_hr_run_85_hrmax, 
        f_rr_run_85_hrmax, 
        f_ve_run_85_hrmax,
        f_hr_run_85_hrmax_age, 
        f_rr_run_85_hrmax_age, 
        f_ve_run_85_hrmax_age,
    ],axis=1)
    vo2_max = max(calculate_moving_average(df_idx)['VO2'])/info_idx['Weight'].values[0]

    if ~np.any(features_all.isnull()):
        features_df[id_test] = [
            info_idx['Age'].values[0], info_idx['Weight'].values[0], info_idx['Height'].values[0], info_idx['Sex'].values[0],
        ] +  features_all.values.tolist()[0] + [vo2_max]
        info_final.append(info_idx)
        patients_id.append(info_idx['ID'].values[0])
features_df = pd.DataFrame(features_df, index=['Age', 'Weight', 'Hight', 'Sex'] + features_all.columns.tolist() + ['VO2_max']).T


#%% Preparing datasets
# VO2peak
columns_info = ['Age', 'Weight', 'Hight', 'Sex']
columns_warmup = columns_info + f_hr_warmup.columns.tolist()
columns_warmup_resp = columns_info + f_hr_warmup.columns.tolist() + f_ve_warmup.columns.tolist() + f_rr_warmup.columns.tolist() 

columns_run_85_hrmax = columns_info + f_hr_run_85_hrmax.columns.tolist() 
columns_run_85_hrmax_age = columns_info + f_hr_run_85_hrmax_age.columns.tolist() 

columns_run_resp_85_hrmax = columns_run_85_hrmax + f_rr_run_85_hrmax.columns.tolist()  + f_ve_run_85_hrmax.columns.tolist() 
columns_run_resp_85_hrmax_age = columns_run_85_hrmax_age + f_rr_run_85_hrmax_age.columns.tolist()  + f_ve_run_85_hrmax_age.columns.tolist() 

columns_warmup_run_85_hrmax = columns_warmup + f_hr_run_85_hrmax.columns.tolist() 
columns_warmup_run_85_hrmax_age = columns_warmup + f_hr_run_85_hrmax_age.columns.tolist() 

columns_warmup_run_resp_85_hrmax = columns_warmup_run_85_hrmax + f_ve_warmup.columns.tolist() + f_rr_warmup.columns.tolist() + f_rr_run_85_hrmax.columns.tolist()  + f_ve_run_85_hrmax.columns.tolist() 
columns_warmup_run_resp_85_hrmax_age = columns_warmup_run_85_hrmax_age + f_ve_warmup.columns.tolist() + f_rr_warmup.columns.tolist() + f_rr_run_85_hrmax_age.columns.tolist()  + f_ve_run_85_hrmax_age.columns.tolist() 


X_demografia = features_df[columns_info] 
X_warmup = features_df[columns_warmup] 
X_warmup_resp = features_df[columns_warmup_resp] 

X_run_85_hrmax = features_df[columns_run_85_hrmax]
X_run_85_hrmax_age = features_df[columns_run_85_hrmax_age]

X_run_resp_85_hrmax = features_df[columns_run_resp_85_hrmax]
X_run_resp_85_hrmax_age = features_df[columns_run_resp_85_hrmax_age]

X_warmup_run_85_hrmax = features_df[columns_warmup_run_85_hrmax]
X_warmup_run_85_hrmax_age = features_df[columns_warmup_run_85_hrmax_age]

X_warmup_run_resp_85_hrmax = features_df[columns_warmup_run_resp_85_hrmax]
X_warmup_run_resp_85_hrmax_age = features_df[columns_warmup_run_resp_85_hrmax_age]

y = features_df['VO2_max']


X_list = [
    X_demografia, 
    X_warmup, 
    X_warmup_resp, 
    X_run_85_hrmax, 
    X_run_85_hrmax_age,
    X_run_resp_85_hrmax, 
    X_run_resp_85_hrmax_age,
    X_warmup_run_85_hrmax, 
    X_warmup_run_85_hrmax_age,
    X_warmup_run_resp_85_hrmax, 
    X_warmup_run_resp_85_hrmax_age
]
    
X_names = [
    'DEMOGRAPHY',
    'WARMUP', 
    'WARMUP_RESP', 
    'RUN_85_HRMAX', 
    'RUN_85_HRMAX_AGE',
    'RUN_RESP_85_HRMAX', 
    'RUN_RESP_85_HRMAX_AGE',
    'WARMUP_RUN_85_HRMAX', 
    'WARMUP_RUN_85_HRMAX_AGE',
    'WARMUP_RUN_RESP_85_HRMAX', 
    'WARMUP_RUN_RESP_85_HRMAX_AGE',
]


#%% RF
dataset_version = 'VO2peak'
model_type = 'RF'

for n_estimators in [50, 100, 1000]:
    for max_depth in [3,4,5,6]:
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
            }
        fitting_params = {}
        for idx in range(len(X_list)):
            X = X_list[idx]
            X_name = X_names[idx]
            _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
            time.sleep(1)

#%% SVR
dataset_version = 'VO2peak'
model_type = 'SVR'

for kernel in ['rbf', 'poly']:
    for c in [10, 1, 0.1, 0.01]:
        for epsilon in [1, 0.1, 0.01]:
            model_params = {
                'kernel': kernel,
                'C': c,
                'epsilon': epsilon
                }
            fitting_params = {}
            for idx in range(len(X_list)):
                X = X_list[idx]
                X_name = X_names[idx]
                _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
                time.sleep(1)
#%% Lasso
dataset_version = 'VO2peak'
for alpha in [0.5,0.3,0.2,0.15, 0.12, 0.1, 0.07, 0.05]:
    model_params = {'alpha': alpha}
    fitting_params = {}
    model_type = 'Lasso'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
        time.sleep(1)
#%% Ridge
dataset_version = 'VO2peak'
for alpha in [100, 50, 25, 10, 1, 0.1, 0.01]:
    model_params = {'alpha': alpha}
    fitting_params = {}
    model_type = 'Ridge'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)
        time.sleep(1)
#%% BayesianRidge

dataset_version = 'VO2peak'
for alpha1 in [1e-5, 1e-6, 1e-7]:
    for alpha2 in [1e-5, 1e-6, 1e-7]:
        for lambda1 in [1e-5, 1e-6, 1e-7]:
            for lambda2 in [1e-5, 1e-6, 1e-7]:

                model_params = {
                    'alpha_1': alpha1,
                    'alpha_2': alpha2,
                    'lambda_1': lambda1,
                    'lambda_2': lambda2,
                    }
                fitting_params = {}
                model_type = 'BayesianRidge'
                for idx in range(len(X_list)):
                    X = X_list[idx]
                    X_name = X_names[idx]
                    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% ARDRegression
dataset_version = 'VO2peak'
for alpha1 in [1e-5, 1e-6, 1e-7]:
    for alpha2 in [1e-5, 1e-6, 1e-7]:
        for lambda1 in [1e-5, 1e-6, 1e-7]:
            for lambda2 in [1e-5, 1e-6, 1e-7]:

                model_params = {
                    'alpha_1': alpha1,
                    'alpha_2': alpha2,
                    'lambda_1': lambda1,
                    'lambda_2': lambda2,
                    }
                fitting_params = {}
                model_type = 'ARDRegression'
                for idx in range(len(X_list)):
                    X = X_list[idx]
                    X_name = X_names[idx]
                    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% GradientBoostingRegressor

dataset_version = 'VO2peak'
for lr in [0.1, 0.01]:
    model_params = {
        'learning_rate': lr
        }
    fitting_params = {}
    model_type = 'GradientBoostingRegressor'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% HuberRegressor
dataset_version = 'VO2peak'
for epsilon in [1.1, 1.2, 1.3]:
    model_params = {
        'epsilon': epsilon,
        'max_iter':10000
        }
    fitting_params = {}
    model_type = 'HuberRegressor'
    for idx in range(len(X_list)):
        X = X_list[idx]
        X_name = X_names[idx]
        _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% TheilSenRegressor

dataset_version = 'VO2peak'
model_params = {
    'max_iter':10000
    }
fitting_params = {}
model_type = 'TheilSenRegressor'
for idx in range(len(X_list)):
    X = X_list[idx]
    X_name = X_names[idx]
    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)

#%% GaussianProcessRegressor
dataset_version = 'VO2peak'

model_params = {
    }
fitting_params = {}
model_type = 'GaussianProcessRegressor'
for idx in range(len(X_list)):
    X = X_list[idx]
    X_name = X_names[idx]
    _, _, _, _ = run_cv(X, y, X_name, model_type, ModelSKL, model_params, fitting_params, dataset_version)


#%% MLP
dataset_version = 'VO2peak'
model_type = 'MLP'
for neurons in [[25], [50], [50, 25]]:
    for alpha in [1,0.1, 0.01]:
        for learning_rate in [0.1, 0.01, 0.001]:
            model_params = {
                'neurons': neurons, 
                'batch_norm': True, 
                'dropout': 0,
                'regularization': 'l1',
                'regularization_alpha': alpha
            }
            fitting_params = {
                'learning_rate' : learning_rate, 
                'epochs': 1000, 
                'batch_size': 32, 
                'decay_steps': 10, 
                'decay_rate': 0.95
            }
            for idx in range(len(X_list)):
                X = X_list[idx]
                X_name = X_names[idx]
                _, _, _, _ = run_cv(X, y, X_name, model_type, ModelMLP, model_params, fitting_params, dataset_version)
