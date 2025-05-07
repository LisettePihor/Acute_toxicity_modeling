import pandas as pd
import pickle
import os
from Model_creation import training_and_test, plot_prediciton
import tensorflow as tf
SEED = 42
tf.random.set_seed(SEED)
import keras
from contextlib import redirect_stdout

#prepare data
data = pd.read_csv('Tetrahymena_rdkit.csv', index_col='Unnamed: 0')
X = data.drop('Dependent', axis=1)
y = data['Dependent']
X_train,y_train,X_test,y_test = training_and_test(X,y, 0.90, 0.01)
print(len(X_train))
print(len(X_test))
print(len(X_train.columns))
#choose which trial to look
path = 'main_100'
#get the model with all features and save the model summary as a table
model = keras.models.load_model(os.path.join(path,'model.keras'))
with open(os.path.join(path,'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()
#create dataframe from all of the selection methods
functions = ['L1', 'permutation','tree','VIANN']
df = pd.DataFrame()
#for each function sort the dataframe, 
#save the R2, R2 test, features, model and function for the first five best models as a csv file 

#and add the function dataframe to the big dataframe
for function in functions:
    function_file = os.path.join(path,f'{function}.pkl')
    if os.path.exists(function_file):
        with open(function_file, 'rb') as file:
            best_models_list = pickle.load(file)
        best_models_df = pd.DataFrame(best_models_list, columns=['R2','R2_test','model','params','features','function'])
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
df_few = df[df['nr_features'] < 15] 

best_features = pd.DataFrame()
#get best two models
for i in range(2):
    best = df.iloc[i]
    best_few = df_few.iloc[i]
    #plot prediction for the best two models
    best_model = best['model']
    best_model_few = best_few['model']

    X_train_new = X_train[best['features']]
    X_test_new = X_test[best['features']]
    pred_train = best_model.predict(X_train_new)
    pred_test = best_model.predict(X_test_new)
    plot_prediciton(os.path.join(path,f'best_{i+1}_prediction.svg'),y_train,y_test,pred_train,pred_test)

    X_train_new_few = X_train[best_few['features']]
    X_test_new_few = X_test[best_few['features']]
    pred_train_few = best_model_few.predict(X_train_new_few)
    pred_test_few = best_model_few.predict(X_test_new_few)
    plot_prediciton(os.path.join(path,f'best_{i+1}_few_prediction.svg'),y_train,y_test,pred_train_few,pred_test_few)
    #collect the features for all to a dataframe
    best_features[f'{i+1}_model'] = [best['features']]
    best_features[f'{i+1}_model_few'] = [best_few['features']]

    #save the model summary for both
    with open(os.path.join(path,f'best_{i+1}_modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            best_model.summary()
    with open(os.path.join(path,f'best_{i+1}_few_modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            best_model_few.summary()

#save the features dataframe
best_features.to_csv(os.path.join(path,f'best_features.csv'), index=False)
# Save the big dataframe to CSV
df.drop(columns=['model'], inplace=True)
df.to_csv(os.path.join(path,f'all_best_models.csv'), index=False)
df_few.drop(columns=['model'], inplace=True)
df_few.to_csv(os.path.join(path,f'best_models_few.csv'), index=False)