import pandas as pd
import pickle
import os
from Model_creation import training_and_test, plot_prediciton
import tensorflow as tf
SEED = 42
tf.random.set_seed(SEED)
import keras
from contextlib import redirect_stdout
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

#choose which trial to look
path = 'main_sampler'
#choose which functions to look
functions = ['L1','tree','VIANN','permutation']
#prepare data
data = pd.read_csv('Tetrahymena_rdkit.csv', index_col='Unnamed: 0')
X = data.drop('Dependent', axis=1)
y = data['Dependent']
X_train,y_train,X_test,y_test = training_and_test(X,y, 0.90, 0.01)
kf = KFold(n_splits=10, shuffle=True, random_state=101)
print(len(X_train))
print(len(X_test))
print(len(X_train.columns))
#get the model with all features and save the model summary as a table
model = keras.models.load_model(os.path.join(path,'model.keras'))
with open(os.path.join(path,'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()
#create dataframe from all of the selection methods
df = pd.DataFrame()

'''A helpful function for calculating RMSE'''
def get_RMSE(row,X,y,type):
    model = row['model']
    features = row['features']
    X_test = X[features]
    if type=='test':
        y_pred=model.predict(X_test)
        rmse = root_mean_squared_error(y,y_pred)
    elif type == 'train':
        rmse_list = list()
        for train_index, test_index in kf.split(X_test):
            X_train_kf, X_test_kf = X_test.iloc[train_index], X_test.iloc[test_index]
            y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
            y_pred= model.predict(X_test_kf)
            rmse = root_mean_squared_error(y_test_kf, y_pred)
            rmse_list.append(rmse)
        rmse= np.mean(rmse_list)
    return rmse
#save the R2, R2 test, RMSE, RMSE test, features and function for the models as a csv file 
#and add the function dataframe to the big dataframe
for function in functions:
    function_file = os.path.join(path,f'{function}.pkl')
    if os.path.exists(function_file):
        with open(function_file, 'rb') as file:
            best_models_list = pickle.load(file)
        best_models_df = pd.DataFrame(best_models_list, columns=['R2','R2_test','model','params','features','function'])
        best_models_df['RMSE'] = best_models_df.apply(lambda row: get_RMSE(row,X_train,y_train,'train'),axis=1)
        best_models_df['RMSE_test'] = best_models_df.apply(lambda row: get_RMSE(row,X_test,y_test,'test'),axis=1)
        best_models_df['nr_features'] = best_models_df['features'].apply(len)
        
        plt.plot(best_models_df['nr_features'], best_models_df['R2'], marker='o',label=f'{function}')
        best_models_df = best_models_df.sort_values(by='R2', ascending=False)
        # Add to the big dataframe
        df = pd.concat([df, best_models_df], ignore_index=True)
        # Drop column with model before saving the DataFrame
        best_models_df = best_models_df.drop(columns=['model'])
        # Save the best models to CSV
        best_models_df.to_csv(os.path.join(path,f'{function}_best_models.csv'), index=False)

#Create a plot for comparing the functions
features_plot = os.path.join(path,'comparison.svg')
plt.title('Tunnuste arv ja R²')
plt.xlabel('Tunnuste arv')
plt.ylabel('R²')
plt.legend()
plt.savefig(features_plot)
plt.close()

#Sort the big dataframe
df = df.sort_values(by='nr_features', ascending=True)
best = df.iloc[0]
#plot prediction for the best model
best_model = best['model']
X_train_new = X_train[best['features']]
X_test_new = X_test[best['features']]
pred_train = best_model.predict(X_train_new)
pred_test = best_model.predict(X_test_new)
plot_prediciton(os.path.join(path,f'best_prediction.svg'),y_train,y_test,pred_train,pred_test)
#save the model summary for both
with open(os.path.join(path,f'best_modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        best_model.summary()
# Save the big dataframe to CSV
df.drop(columns=['model'], inplace=True)
df.to_csv(os.path.join(path,f'all_best_models.csv'), index=False)