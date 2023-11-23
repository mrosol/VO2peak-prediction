import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import tensorflow as tf
from tqdm import tqdm
import neptune
import os
from datetime import datetime
from pyCompare import blandAltman
import logging

logging.basicConfig(level=logging.INFO)

def select_rows_greater_than_previous(df):
    selected_rows = []
    max_speed = float('-inf')  # Initialize with negative infinity
    
    for _, row in df.iterrows():
        if row['Speed'] >= max_speed:
            selected_rows.append(row)
            max_speed = row['Speed']
    
    selected_df = pd.DataFrame(selected_rows)
    return selected_df

def select_rows(df, hr_threshold, hr_max):
    for i in range(len(df)):
        if df.iloc[i,:]['HR'] > hr_threshold*hr_max:
            return df.iloc[:i]
    return df


def calculate_moving_average(df):
    window_size = 15
    average_values_vo2 = []
    average_values_vco2 = []
    average_values_ve = []
    average_values_speed = []
    time_points = []
    df = select_rows_greater_than_previous(df)
    for i in range(len(df) - window_size + 1):
        subset = df.iloc[i:i+window_size]
        avg_vo2 = subset['VO2'].mean()
        avg_vco2 = subset['VCO2'].mean()
        avg_ve = subset['VE'].mean()
        avg_speed = subset['Speed'].mean()
        average_values_vo2.append(avg_vo2)
        average_values_vco2.append(avg_vco2)
        average_values_ve.append(avg_ve)
        average_values_speed.append(avg_speed)
        time_point = subset.iloc[7]['time']  # Assuming time is a column name
        time_points.append(time_point)
    new_df = pd.DataFrame({'Time': time_points, 'VO2': average_values_vo2, 'VCO2': average_values_vco2, 'VE': average_values_ve, 'Speed': average_values_speed})
    return new_df

def calc_feature(time, series, name):
    mean = np.mean(series)
    std = np.std(series)
    min_val = np.min(series)
    max_val = np.max(series)
    median = np.median(series)
    q25 = np.quantile(series, 0.25)
    q75 = np.quantile(series, 0.75)
    skew = stats.skew(series)
    kurt = stats.kurtosis(series)
    impuls_factor = max_val/mean
    max_min = max_val-min_val
    try:
        lr = LinearRegression().fit(time.values.reshape(-1, 1), series.values.reshape(-1, 1))
        alpha = lr.coef_
    except:
        alpha = np.nan
    shape_factor = np.sqrt(np.mean(series**2))/np.mean(abs(series))
    features = ['mean', 'std', 'minimum', 'maximum', 'median', 'q25', 'q75', 'skewness', 'kurtosis', 'impuls_factor', 'shape_factor', 'max_min', 'lr_alpha']
    columns = []
    for feat in features:
        columns.append(f'{name}_{feat}')
    return pd.DataFrame(
        [mean, std, min_val, max_val, median, q25, q75, skew, kurt, impuls_factor, shape_factor, max_min, alpha],
        index=columns
    ).T

def calcualte_all_features(df_idx_stage, stage):
    features_hr_stage = calc_feature(df_idx_stage['time'], df_idx_stage['HR'], f'HR_{stage}')
    features_rr_stage = calc_feature(df_idx_stage['time'], df_idx_stage['RR'], f'RR_{stage}')
    features_ve_stage = calc_feature(df_idx_stage['time'], df_idx_stage['VE'], f'VE_{stage}')

    return features_hr_stage, features_rr_stage, features_ve_stage

class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
            
class ModelMLP():
    def __init__(self, model_type, neurons, batch_norm, dropout, regularization=None, regularization_alpha=0.1):

        if regularization=='l1':
            kernel_regularizer = tf.keras.regularizers.L1(
                    l1=regularization_alpha
                )
        elif regularization=='l2':
            kernel_regularizer = tf.keras.regularizers.L2(
                    l2=regularization_alpha
                )
        else:
            kernel_regularizer = None

        mdl = Sequential()
        for neur in neurons:
            mdl.add(Dense(neur, activation='elu', kernel_regularizer=kernel_regularizer))
            if batch_norm:
                mdl.add(BatchNormalization())
            if dropout>0:
                mdl.add(Dropout(dropout))
        mdl.add(Dense(1, activation='linear'))
        self.model = mdl
        self.params = {
            'neurons': str(neurons)[1:-1],
            'batch_norm': batch_norm,
            'dropout': dropout
        }


    def fit(self, X_train_scaled, y_train, folder_name, idx,learning_rate, epochs, batch_size, decay_steps, decay_rate):
        opt = optimizers.Nadam(learning_rate=learning_rate)
        self.model.compile( 
            optimizer=opt,
            loss=tf.keras.losses.MeanSquaredError(), #"binary_crossentropy",
            metrics = [
                metrics.MeanSquaredError(name='mse'),
                metrics.MeanAbsoluteError(name='mae'),
                metrics.MeanAbsolutePercentageError(name='mape'),
                metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        # earlystop = ReturnBestEarlyStopping(
        #     monitor=f"val_mape",
        #     min_delta=0.05,
        #     patience=100,
        #     verbose=0,
        #     mode="auto",
        #     baseline=None,
        #     restore_best_weights=True,
        #     # start_from_epoch = 0
        #     )
        # X_train_scaled, X_validation_scaled, y_train, y_valiation = train_test_split(X_train_scaled, y_train, test_size=1/9, shuffle=True)
        # X_train_scaled = pd.concat([X_train_scaled,X_train_scaled])
        # y_train = pd.concat([y_train,y_train])
        hist = self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_callback], verbose=False)  #, validation_data=(X_validation_scaled, y_valiation) earlystop
        self.model.save(f'{folder_name}/model_{idx}.h5')
        for metric in ['loss', 'mse', 'mae', 'mape', 'rmse', 'lr']:
            plt.figure(f'{folder_name}_{metric}')
            plt.plot(hist.history[metric])
            # if metric != 'lr':
            #     plt.plot(hist.history[f'val_{metric}'], '--', alpha=0.3)
        
        self.params['epochs'] = epochs
        self.params['batch_size'] = batch_size
        self.params['learning_rate'] = learning_rate

        return hist 
    
    def predict(self, X):
        return self.model.predict(X)

    def finished_cv(self, folder_name):
        for metric in ['loss', 'mse', 'mae', 'mape', 'rmse', 'lr']:
            fig = plt.figure(f'{folder_name}_{metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            fig.savefig(f'{folder_name}/{metric}.jpg')

class ModelSKL():

    def __init__(self, model_type, **model_params):

        if model_type=='LR':
            mdl = LinearRegression()
        elif model_type=='Lasso':
            mdl = Lasso(**model_params)
        elif model_type=='Ridge':
            mdl = Ridge(**model_params)
        elif model_type=='SVR':
            mdl = SVR(**model_params)
        elif model_type=='RF':
            mdl = RandomForestRegressor(**model_params)
        elif model_type=='BayesianRidge':
            mdl = BayesianRidge(**model_params)
        elif model_type=='ARDRegression':
            mdl = ARDRegression(**model_params)
        elif model_type=='GaussianProcessRegressor':
            mdl = GaussianProcessRegressor(**model_params)
        elif model_type=='GradientBoostingRegressor':
            mdl = GradientBoostingRegressor(**model_params)
        elif model_type=='HuberRegressor':
            mdl = HuberRegressor(**model_params)
        elif model_type=='TheilSenRegressor':
            mdl = TheilSenRegressor(**model_params)
        #TODO other models
        self.model = mdl
        self.params = {}
        for key in model_params:
            self.params[key] = model_params[key]

    def fit(self, X_train_scaled, y_train, folder_name, idx, **fitting_params):
        self.model.fit(X_train_scaled, y_train, **fitting_params)

        joblib.dump(self.model, f'{folder_name}/model_{idx}.joblib')
        for key in fitting_params:
            self.params[key] = fitting_params[key]

    def predict(self, X):
        return self.model.predict(X)
    
    def finished_cv(self, folder_name):
        pass



def run_cv(X, y, signals, model_type, model_class, model_params, fitting_params, dataset_version):
    PROJECT_NAME = ''
    API_TOKEN = ''
    
    X['Sex'][X['Sex']==0] = -1
    kf = KFold(n_splits=10, random_state=123, shuffle=True)

    folder_name = f'Artifacts/Artifacts_{model_type}_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}'
    os.makedirs(folder_name)
    
    y_hat_test_all = np.array([])
    y_test_all = np.array([])
    y_hat_train_all = np.array([])
    y_train_all = np.array([])
    sex_train = np.array([])
    sex_test = np.array([])
    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []
    mae_train = []
    mae_test = []
    mape_train = []
    mape_test = []    
    
    for idx, (train_index, test_index) in tqdm(enumerate(kf.split(X, y)),total=10):

        X_train, X_test  =  X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test  =  y.iloc[train_index,], y.iloc[test_index,]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)    

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        X_train_scaled['Sex'] = X_train['Sex'].values
        X_test_scaled['Sex'] = X_test['Sex'].values

        mdl = model_class(model_type, **model_params)

        mdl.fit(X_train_scaled, y_train, folder_name, idx, **fitting_params)

        y_hat_train = mdl.predict(X_train_scaled)
        y_hat_test = mdl.predict(X_test_scaled)
        y_hat_test_all = np.concatenate([y_hat_test_all, y_hat_test.reshape(-1)])
        y_test_all = np.concatenate([y_test_all, y_test.values])
        y_hat_train_all = np.concatenate([y_hat_train_all, y_hat_train.reshape(-1)])
        y_train_all = np.concatenate([y_train_all, y_train.values])
        sex_train = np.concatenate([sex_train, X_train['Sex'].values])
        sex_test = np.concatenate([sex_test, X_test['Sex'].values])
        r2_train.append(r2_score(y_train, y_hat_train))
        r2_test.append(r2_score(y_test, y_hat_test))
        mse_train.append(mean_squared_error(y_train, y_hat_train))
        mse_test.append(mean_squared_error(y_test, y_hat_test))
        mae_train.append(mean_absolute_error(y_train, y_hat_train))
        mae_test.append(mean_absolute_error(y_test, y_hat_test))
        mape_train.append(np.mean(np.abs(y_train-y_hat_train.reshape(-1))/y_train)*100)
        mape_test.append(np.mean(np.abs(y_test-y_hat_test.reshape(-1))/y_test)*100)

    mdl.finished_cv(folder_name)

    fig_y_ypred = plt.figure()
    min_val = (min(np.min(y_test_all), np.min(y_hat_test_all)) // 10) * 10 
    max_val = (max(np.max(y_test_all), np.max(y_hat_test_all)) // 10) * 10 + 10
    plt.plot(y_test_all[sex_test==-1], y_hat_test_all[sex_test==-1], 'o', alpha=0.4, label='Man')
    plt.plot(y_test_all[sex_test==1], y_hat_test_all[sex_test==1], 'o', alpha=0.4, label='Woman')
    plt.xlabel('True $VO_{2max}$')
    plt.ylabel('Predicted $VO_{2max}$')
    plt.xticks(np.arange(min_val, max_val+1, 10))
    plt.yticks(np.arange(min_val, max_val+1, 10))
    plt.legend()
    fig_y_ypred.savefig(f'{folder_name}/y_vs_ypred.jpg')

    fig_y_ypred_train = plt.figure()
    min_val = (min(np.min(y_train_all), np.min(y_hat_train_all)) // 10) * 10
    max_val = (max(np.max(y_train_all), np.max(y_hat_train_all)) // 10) * 10 + 10
    plt.plot(y_train_all[sex_train==-1], y_hat_train_all[sex_train==-1], 'o', alpha=0.2, label='Man')
    plt.plot(y_train_all[sex_train==1], y_hat_train_all[sex_train==1], 'o', alpha=0.2, label='Woman')
    plt.xlabel('True $VO_{2max}$')
    plt.ylabel('Predicted $VO_{2max}$')
    plt.xticks(np.arange(min_val, max_val+1, 10))
    plt.yticks(np.arange(min_val, max_val+1, 10))
    plt.legend()
    fig_y_ypred_train.savefig(f'{folder_name}/y_vs_ypred_train.jpg')

    blandAltman(y_hat_test_all, y_test_all,
            limitOfAgreement=1.96,
            confidenceInterval=95,
            confidenceIntervalMethod='approximate',
            detrend=None,
            percentage=False,
            savePath=f'{folder_name}//ba.jpg')
    
    res_train = pd.DataFrame({
        'MAPE': mape_train,
        'R2': r2_train,
        'MAE': mae_train,
        'MSE': mse_train
    })
    res_test = pd.DataFrame({
        'MAPE': mape_test,
        'R2': r2_test,
        'MAE': mae_test,
        'MSE': mse_test
    })
    results  = {'train': res_train, 'test': res_test}
    res_train.to_csv(f'{folder_name}/train_metrics.csv')
    res_test.to_csv(f'{folder_name}/test_metrics.csv')
    pd.DataFrame({
        'y_true': y_test_all,
        'y_pred': y_hat_test_all
        }).to_csv(f'{folder_name}/y_true_pred.csv')
    # Logging to Neptune
    params = mdl.params
    params['dataset_version'] = dataset_version
    params['model_type'] = model_type

    run = neptune.init_run(
        project=PROJECT_NAME,
        api_token=API_TOKEN,
    ) 
    run["parameters"] = params
    for stage in ['train', 'test']:
        for metric in ['MAPE', 'R2', 'MAE', 'MSE']:
            run[f'{stage}/{metric}'] = results[stage].mean()[metric]
    run["Artifacts"].upload_files(f"{folder_name}/*")  
    run["sys/tags"].add([signals])
    run.stop()
    logging.info(f"MAPE={results['test'].mean()['MAPE']}")

    # Deleting all artifacts
    for filename in os.listdir(folder_name):
        filepath = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
    os.rmdir(folder_name) 
    plt.close('all')

    return y_hat_test_all, y_test_all, results, fig_y_ypred
