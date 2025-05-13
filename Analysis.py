import pandas as pd
import pickle
import os
from Model_creation import training_and_test
import tensorflow as tf
SEED = 42
tf.random.set_seed(SEED)
import keras
from contextlib import redirect_stdout
from sklearn.metrics import root_mean_squared_error

#prepare data
data = pd.read_csv('Tetrahymena_rdkit.csv', index_col='Unnamed: 0')
X = data.drop('Dependent', axis=1)
y = data['Dependent']
X_train,y_train,X_test,y_test = training_and_test(X,y, 0.90, 0.01)
print(len(X_train))
print(len(X_test))
print(len(X_train.columns))
#choose which trial to look
path = 'main'
#get the model with all features and save the model summary as a table
model = keras.models.load_model(os.path.join(path,'model.keras'))
with open(os.path.join(path,'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()
#create dataframe from all of the selection methods
functions = ['L1', 'permutation','tree','VIANN']
df = pd.DataFrame()
#for each function sort the dataframe, 
#save the R2, R2 test, RMSE, RMSE test, features and function for the first five best models as a csv file 
def get_RMSE(row,X,y):
    model = row['model']
    features = row['features']
    X_test = X[features]
    y_pred=model.predict(X_test)
    rmse = root_mean_squared_error(y,y_pred)
    return rmse
#and add the function dataframe to the big dataframe
for function in functions:
    function_file = os.path.join(path,f'{function}.pkl')
    if os.path.exists(function_file):
        with open(function_file, 'rb') as file:
            best_models_list = pickle.load(file)
        best_models_df = pd.DataFrame(best_models_list, columns=['R2','R2_test','model','params','features','function'])
        best_models_df['RMSE_test'] = best_models_df.apply(lambda row: get_RMSE(row,X_test,y_test),axis=1)
        best_models_df['nr_features'] = best_models_df['features'].apply(len)
        
        
        best_models_df = best_models_df.sort_values(by='R2', ascending=False)
        # Add to the big dataframe
        df = pd.concat([df, best_models_df], ignore_index=True)
        # Drop column with model before saving the DataFrame
        best_models_df = best_models_df.drop(columns=['model'])
        # Save the best models to CSV
        best_models_df.to_csv(os.path.join(path,f'{function}_best_models.csv'), index=False)

#Sort the big dataframe
df = df.sort_values(by='R2', ascending=False)

# Save the big dataframe to CSV
df.drop(columns=['model'], inplace=True)
df.to_csv(os.path.join(path,f'all_best_models.csv'), index=False)