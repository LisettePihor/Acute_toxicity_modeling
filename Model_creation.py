import pickle
import numpy as np
import optuna
import keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'
import tensorflow as tf
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import pandas as pd
from custom_callback import VarImpVIANN
from sklearn.model_selection import train_test_split, KFold
SEED = 42
tf.random.set_seed(SEED)


#optuna objective method
def objective(trial,input_shape_obj, X_train,y_train,kf):
    """Optuna objective function"""
    n_layers = trial.suggest_int("n_layers", 3, 10)
    n_epochs = trial.suggest_int('n_epochs', 50*n_layers, 500*n_layers)
    batch_size = trial.suggest_int('batch_size', 100, 400)
    patience = trial.suggest_int('patience',0,4)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape_obj))
    for i in range(n_layers):
        num_neurons = trial.suggest_int("n_units_l{}".format(i), 16, 1000, log=True)
        if i == 0: 
            model.add(keras.layers.Dense(units=num_neurons,activation='relu',kernel_regularizer=keras.regularizers.L1(0.02)))
        else: 
            model.add(keras.layers.Dense(units=num_neurons,activation='relu'))
            model.add(keras.layers.Dropout(rate=0.3))
    
    model.add(keras.layers.Dense(units=1,activation='linear'))
    learning_rate = trial.suggest_float("learn_rate", 0.0001, 0.01, log=True)
    optimizer= keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['r2_score'])
    model.fit(X_train, y_train, 
                batch_size=batch_size, 
                verbose=0, 
                epochs=n_epochs,
                callbacks=[early_stopping])
    rmse = evaluate_model_KFold_rmse(model,X_train,y_train,kf)
    return rmse

#Data preparation functions
def remove_correlated(df:pd.DataFrame,threshold:float):
    corr_matrix = df.corr().abs()
    to_keep = []
    for feature in corr_matrix:
        if feature not in to_keep:
            if all(corr_matrix[feature][to_keep] < threshold):
                to_keep.append(feature)
    new_df = df[to_keep]
    return new_df

def training_and_test(X,y,corr_threshold, var_threshold):
    """Splits and scales given data
    Removes correlated and low variance features"""
    #Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Remove correlated
    X_train_corr = remove_correlated(X_train,corr_threshold)
    X_test_corr = X_test[X_train_corr.columns]
    print(f'Removed {len(X.columns)-len(X_train_corr.columns)} features')
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_corr),columns=X_train_corr.columns,index=X_train_corr.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_corr),columns=X_test_corr.columns,index=X_test_corr.index)

    #Remove low variance
    sel = VarianceThreshold(var_threshold)
    X_var_train = pd.DataFrame(sel.fit_transform(X_train_scaled), columns = sel.get_feature_names_out())
    X_var_test = pd.DataFrame(sel.transform(X_test_scaled),columns = sel.get_feature_names_out())

    print(f'Removed {len(X_train_corr.columns)-len(X_var_train.columns)} features')
    return X_var_train,y_train,X_var_test,y_test


#Model creation functions
def model_from_params(path:str, params:dict,input_shape, X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame, 
                      hist_filename:str,VIANN_needed=False):
    """Creates and trains a model from given parameters, when given a file path for VIANN feature importance calclulates
        it if it does not already exist, also creates two training plots"""
    #create the model from the given dictionary
    param_model = keras.Sequential()
    param_model.add(keras.Input(shape=input_shape))
    n_layers= params.get('n_layers')
    for i in range(n_layers):
        n_units = params.get("n_units_l{}".format(i)) 
        if i == 0: 
            param_model.add(keras.layers.Dense(units=n_units,activation='relu',kernel_regularizer=keras.regularizers.L1(0.02)))
        else: 
            param_model.add(keras.layers.Dense(units=n_units,activation='relu'))
            param_model.add(keras.layers.Dropout(0.3))
    param_model.add(keras.layers.Dense(units=1,activation='linear'))
    optimizer = keras.optimizers.Adam(params.get('learn_rate'))
    param_model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=['r2_score'])
    n_epochs=params.get("n_epochs")
    batch_size = params.get("batch_size")
    patience = params.get("patience")
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    callbacks = [early_stopping]
    #Add VIANN if needed
    if VIANN_needed: 
        VIANN = VarImpVIANN(verbose=0)
        callbacks.append(VIANN)
    #Train the model
    history = param_model.fit(X_train, y_train, 
                  batch_size=batch_size, 
                  validation_data=(X_test,y_test),
                  verbose=0, 
                  epochs=n_epochs,
                  callbacks=callbacks)
    #Plot training loss
    history_path = os.path.join(path,hist_filename)
    if not os.path.isfile(history_path):
        plot_history(history_path,history)
    #Save VIANN if needed
    if VIANN_needed:
        VIANN_path = os.path.join(path,'VIANN_scores.pkl')
        VIANN_data = pd.DataFrame({'feature':X_train.columns,'score':VIANN.varScores}).sort_values(by='score')
        with open(VIANN_path,"wb") as f:
                pickle.dump(VIANN_data,f)
    return param_model
def get_model(path:str,nr_trials, data_input_shape, X_train:pd.DataFrame, y_train:pd.DataFrame, 
              X_test:pd.DataFrame, y_test:pd.DataFrame,kf:KFold, pref_filename:str, hist_filename:str,VIANN_needed=False):
    """Checks if the model exists and then either trains a model from scratch or returns existing, saves a plot for prediction"""
    #if path exists, get model and the parameters
    model_path = os.path.join(path,'model.keras')
    params_path = os.path.join(path,'params.pkl')
    if os.path.isfile(model_path):
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
            if VIANN_needed:#If VIANN feature importance wanted but model already trained, train again
                best_model = model_from_params(path, best_params,[data_input_shape],X_train,y_train, X_test,y_test,hist_filename,VIANN_needed)
            else: best_model = keras.models.load_model(model_path)
    #if path does not exist, find optimal parameters with Optuna
    else:
        sampler = optuna.samplers.TPESampler(seed=10)
        study = optuna.create_study(direction='minimize',sampler=sampler)
        study.optimize(lambda trial: objective(trial,[data_input_shape], X_train,y_train,kf), 
                       n_trials=nr_trials,n_jobs=1)
        best_params = study.best_params
        with open(params_path, 'wb') as f:
            pickle.dump(best_params, f)
        best_model = model_from_params(path, best_params,[data_input_shape],X_train,y_train, X_test, y_test,hist_filename,VIANN_needed)
        best_model.save(model_path)

    #show model R2
    pred_train = best_model.predict(X_train)
    pred_test = best_model.predict(X_test)
    print(evaluate_model_KFold(best_model,X_train,y_train,kf))
    print(r2_score(y_test,pred_test))
    print(root_mean_squared_error(y_train,pred_train))
    print(root_mean_squared_error(y_test,pred_test))

    #Plot pred vs true for train and test
    plot = os.path.join(path,pref_filename)
    plot_names = os.path.join(path,f'names_{pref_filename}')
    if not os.path.isfile(plot):
        plot_prediciton(plot,y_train,y_test,pred_train,pred_test)
    if not os.path.isfile(plot_names):
        plot_prediciton(plot_names,y_train,y_test,pred_train,pred_test,True)
    return best_model, best_params 

def plot_prediciton(file_path,y_train,y_test,pred_train,pred_test,names=False):
    #Plot pred vs true for train and test
    plt.scatter(y_train,pred_train, label='Treeningandmed')
    plt.scatter(y_test,pred_test, label="Testandmed")
    if names:
        for i, (true_val, pred_val) in enumerate(zip(y_test, pred_test)):
            plt.annotate(f'{i}', (true_val, pred_val), textcoords="offset points", xytext=(0,5), ha='center')

    plt.title("Tõelised vs ennustatud väärtused")
    plt.xlabel("Tõelised väärtused")
    plt.ylabel("Ennustatud väärtused")
    plt.legend()
    plt.savefig(file_path)
    plt.close()

def plot_history(file_path, history):
    plt.plot(history.history['loss'], label=f'Treening kadu')
    plt.plot(history.history['val_loss'], label=f'Test kadu')
    plt.title(f'Treening ja test kadu')
    plt.xlabel('Epohhid')
    plt.ylabel('Kadu')
    plt.legend()
    plt.savefig(file_path)
    plt.close()
#KFold evaluation methods
#for R2
def evaluate_model_KFold(model,X,y,kf):
    r2_list = list()
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        y_pred= model.predict(X_test_kf)
        r2 = r2_score(y_test_kf, y_pred)
        r2_list.append(r2)
    return np.mean(r2_list)
#for RMSE
def evaluate_model_KFold_rmse(model,X,y,kf):
    rmse_list = list()
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        y_pred= model.predict(X_test_kf)
        rmse = root_mean_squared_error(y_test_kf, y_pred)
        rmse_list.append(rmse)
    return np.mean(rmse_list)