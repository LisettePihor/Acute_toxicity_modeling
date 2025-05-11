import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'
import numpy as np
import pickle
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from Model_creation import evaluate_model_KFold, objective, model_from_params
from math import ceil

#Data visualization
def plot_feature(feature_values,y):
    plt.scatter(feature_values,y)
    plt.title('Tunnuse seos ennustatavaga')
    plt.xlabel('Tunnuste väärtused')
    plt.ylabel('Ennustatava väärtused')
    plt.legend()
    plt.show()
    return None
#Functions for finding the best features using recursive feature elimination
def tree_feature_importance(X_train,y_train):
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    columns = X_train.columns
    coefficients = abs(model.feature_importances_)
    features_df = pd.DataFrame(columns, columns = ['feature'])
    coefficients_df = pd.DataFrame(coefficients, columns = ['coefficient'])
    df = pd.concat((features_df, coefficients_df),axis = 1).sort_values(by='coefficient', ascending = False)
    return df
  
def per_importance(model, X_test, y_test,repeats):
    r = permutation_importance(model,X_test,y_test,n_repeats=repeats,random_state=0,scoring=['r2'])#dictionary
    features = X_test.columns
    for metric in r:
        r_metric = r[metric]
        r_df = pd.DataFrame(data={'feature': features, 'mean': r_metric.get('importances_mean'), 'std':r_metric.get('importances_std')}
                            , columns=['feature','mean','std'])
        r_sorted = r_df.sort_values(by=['mean','std'],ascending=False)
    return r_sorted

def l1_importance(model,current_features):
    weights = model.layers[0].get_weights()[0]
    abs_weights = np.abs(weights).sum(axis=1)
    df = pd.DataFrame({'feature': current_features, 'weights': abs_weights})
    sorted = df.sort_values(by='weights',ascending=False)
    return df

def get_best_features(optuna_trials,function, path, all_features, param_dict, X_train, y_train, X_test,y_test,kf):
    function_file = os.path.join(path,f'{function}.pkl')
    if not os.path.isfile(function_file):
        nr_features = len(all_features)
        current_features = all_features
        best_50=list()
        if function == 'VIANN': 
            VIANN_needed = True
        else: 
            VIANN_needed = False
        threshold = 0.5
        while(nr_features > 3):
            current_features = current_features[:nr_features]
            X_current = X_train[current_features]
            X_test_current = X_test[current_features]
            #Calculate new hyperparameters when nr_features becomes lower than the given threshold
            if nr_features < (len(all_features)*threshold) or nr_features <10:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial,[X_current.shape[1]], X_current,y_train,kf), n_trials=optuna_trials,n_jobs=4)
                param_dict = study.best_params
                threshold = threshold/2
            model = model_from_params(path,param_dict,[X_current.shape[1]],X_current,y_train,X_test_current, y_test, 'History.svg',VIANN_needed)
            r2 = evaluate_model_KFold(model, X_current, y_train,kf)
            #Calculate R2 on the test set
            r2_test = r2_score(y_test,model.predict(X_test_current))
            if function == "permutation":
                features_df = per_importance(model,X_current,y_train,10)
                current_features = features_df["feature"].to_list()
            if function == "tree":
                features_df = tree_feature_importance(X_current,y_train)
                current_features = features_df["feature"].to_list()
            if function == 'L1':
                features_df = l1_importance(model,X_current.columns)
                current_features = features_df["feature"].to_list()
            if function == 'VIANN':
                VIANN_path = os.path.join(path,'VIANN_scores.pkl')
                if os.path.isfile(VIANN_path):
                    with open(VIANN_path,'rb') as f:
                        features_df = pickle.load(f)
                    current_features = features_df["feature"].to_list()
                else: 
                    print('VIANN not found')
                    break
            if(len(best_50)<50):
                best_50.append((r2,r2_test,model,param_dict,current_features,function))
            else:
                best_50.sort(key=lambda x:x[0],reverse=True)
                last = best_50.pop()
                if (last[0] < r2):best_50.append((r2,r2_test,model,param_dict,current_features,function))
                else: best_50.append(last)
            nr_features = nr_features - ceil(nr_features*0.1)
            
        with open(function_file,"wb") as f:
            pickle.dump(best_50,f)
    else:
        with open(function_file,"rb") as f:
            best_50 = pickle.load(f)
    return best_50

def get_features_and_plots(optuna_trials,functions,path,all_features, param_dict, X_train, y_train, X_test,y_test,kf):
    best_50_list = list()
    #if no functions are given return an empty list
    if len(functions) <= 0:
        return best_50_list
    for function in functions:
        best_50features = get_best_features(optuna_trials,function,path,all_features,param_dict,
                                            X_train,y_train,X_test,y_test,kf)
        best_50_list.extend(best_50features)
        #Plot the number of features vs R2
        features_plot = os.path.join(path,f'{function}.svg')
        if not os.path.isfile(features_plot):
                data_sorted = sorted(best_50features, key=lambda x: len(x[4]))
                num_features = [len(features) for _, _, _,_, features,_ in data_sorted]
                r2test_scores = [r2 for _, r2, _, _,_,_ in data_sorted]
                r2train_scores = [r2 for r2, _, _,_,_,_ in data_sorted]
                plt.plot(num_features, r2train_scores,label='Treening andmed', marker='o')
                plt.plot(num_features,r2test_scores,label="Test andmed", marker='o')
                plt.title('Tunnuste arv ja R²')
                plt.xlabel('Tunnuste arv')
                plt.ylabel('R²')
                plt.legend()
                plt.savefig(features_plot)
                plt.close()
    #Add the ones already existing
    all_functions = ['L1', 'permutation','tree','VIANN']
    existing_functions = [column for column in all_functions if column not in functions]
    for function in existing_functions:
        best_50features = get_best_features(optuna_trials,function,path,all_features,param_dict,
                                            X_train,y_train,X_test,y_test,kf)
        best_50_list.extend(best_50features)
    return best_50_list

