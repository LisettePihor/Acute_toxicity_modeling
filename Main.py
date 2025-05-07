from Feature_importance import get_features_and_plots
from Model_creation import get_model, training_and_test
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'
import tensorflow as tf
SEED = 42
tf.random.set_seed(SEED)
import pandas as pd
from sklearn.model_selection import KFold
from sys import argv
import optuna
tf.get_logger().setLevel('ERROR')
optuna.logging.set_verbosity(optuna.logging.ERROR)
#given with arguments: how many trials optuna constructs, the folder where the outcomes should be saved and
#functions to use for feature selection
optuna_rep = int(argv[1])
path = argv[2]
#If the folder does not exist generate it
if not os.path.exists(path):
    os.makedirs(path)
functions = argv[3:]
if 'VIANN' in functions: VIANN_needed = True
else: VIANN_needed = False
#data preprocessing
data = pd.read_csv('Tetrahymena_rdkit.csv', index_col='Unnamed: 0')
X = data.drop('Dependent', axis=1)
y = data['Dependent']
kf = KFold(n_splits=10, shuffle=True, random_state=101)

X_train,y_train,X_test,y_test,X_val,y_val = training_and_test(X,y, 0.90, 0.01)
all_features=X_train.columns
#model creation
model, param_dict = get_model(path,optuna_rep,X_train.shape[1],X_train,y_train,X_test,y_test,
                              kf,'Ennustatud_vs_tõelised.svg','History.svg',VIANN_needed)
#feature selection
best_50_list = get_features_and_plots(optuna_rep,functions,path,all_features,param_dict,X_train,y_train,X_test,y_test,kf)






