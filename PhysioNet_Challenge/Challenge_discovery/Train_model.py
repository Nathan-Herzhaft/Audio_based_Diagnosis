#%%
################################################################################
#
# Libraries and functions
#
################################################################################

from Challenge_discovery.Load_data import *
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt









# %%
################################################################################
#
# Preprocessing and general functions
#
################################################################################

def extract_features_and_labels(data_folder, verbose) :
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_text_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)

    features = list()
    murmurs = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        
    if verbose >= 1:
        print('Done.')
    return features,murmurs


def generate_input(features,murmurs) :
    X = np.vstack(features)
    y = np.vstack(murmurs)

    imputer = SimpleImputer().fit(X)
    X = imputer.transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, murmurs)
    
    return X_train, X_val, y_train, y_val


features,murmurs = extract_features_and_labels('data\\training_data', 1)
X_train, X_val, y_train, y_val = generate_input(features,murmurs)










# %%
################################################################################
#
# Random Forest training and validation
#
################################################################################

def train_Random_Forest(n_estimators,max_leaf_nodes,X_train, y_train) :
    model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,random_state=1)
    model.fit(X_train,y_train)
    return model
    
def accuracy_model(model, X_val, y_val) :
    predictions = model.predict(X_val)
    accuracy = accuracy_score(predictions, y_val)
    return accuracy

def precision_model(model, X_val, y_val) :
    predictions = model.predict(X_val)
    accuracy = precision_score(predictions, y_val)
    return accuracy

def mae_model(model, X_val, y_val) :
    predictions = model.predict(X_val)
    mae = mean_absolute_error(predictions,y_val)
    return mae
    
def compare_Random_Forests_accuracy(n_min,n_max,max_leaf_nodes,X_train, X_val, y_train, y_val, display=True, verbose = False) :
    Acc = []
    for n in range(n_min,n_max) :
        if verbose :
            print(f'{n-n_min+1}/{n_max-n_min}')
        model = train_Random_Forest(n,max_leaf_nodes,X_train,y_train)
        Acc.append(accuracy_model(model,X_val,y_val))
    x = np.linspace(n_min,n_max,n_max-n_min)
    if display :
        plt.plot(x,Acc)
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
    return np.argmax(Acc) + n_min

def compare_Random_Forests_mae(n_min,n_max,max_leaf_nodes,X_train, X_val, y_train, y_val, display=True, verbose = False) :
    MAE = []
    for n in range(n_min,n_max) :
        if verbose :
            print(f'{n-n_min+1}/{n_max-n_min}')
        model = train_Random_Forest(n,max_leaf_nodes,X_train,y_train)
        MAE.append(mae_model(model,X_val,y_val))
    x = np.linspace(n_min,n_max,n_max-n_min)
    if display :
        plt.plot(x,MAE)
        plt.xlabel('n_estimators')
        plt.ylabel('Mean Absolute Error')
    return np.argmin(MAE) + n_min

#Best Accuracy is obtained with n_estimators = 21
#Best Mean Absolute Error is obtained with n_estimators = 49


#%%
model = train_Random_Forest(500,5,X_train,y_train)

accuracy_model(model, X_val, y_val)








# %%
################################################################################
#
# XGBoost training and validation
#
################################################################################

def train_XGBoost(n_estimators,learning_rate, X_train, y_train) :
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,random_state=1)
    model.fit(X_train,y_train,
              early_stopping_rounds=5,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model


def choose_prediction(model,X_val) :
    outputs = model.predict(X_val)
    predictions = []
    for output in outputs :
        label = np.argmax(output)
        predict = [0,0,0]
        predict[label] = 1
        predictions.append(predict)
    predictions = np.vstack(predictions)
    return predictions

def mae_model_preds(model, X_val, y_val) :
    predictions = choose_prediction(model,X_val)
    mae = mean_absolute_error(predictions,y_val)
    return mae

def mae_model_probs(model, X_val, y_val) :
    predictions = model.predict(X_val)
    mae = mean_absolute_error(predictions,y_val)
    return mae

def compare_XGBoosts_probs(lr_min,lr_max,n_estimators,X_train, X_val, y_train, y_val, display=True, verbose = False) :
    MAE = []
    for k in range(10) :
        if verbose :
            print(f'{k+1}/10')
        model = train_XGBoost(n_estimators,lr_min + (k/10)*(lr_max-lr_min),X_train,y_train)
        MAE.append(mae_model_probs(model,X_val,y_val))
    x = np.linspace(lr_min,lr_max,10)
    if display :
        plt.plot(x,MAE)
        plt.xlabel('learning rate')
        plt.ylabel('Mean Absolute Error')
    return (np.argmin(MAE)/10)*(lr_max-lr_min) + lr_min

def compare_XGBoosts_preds(lr_min,lr_max,n_estimators,X_train, X_val, y_train, y_val, display=True, verbose = False) :
    MAE = []
    for k in range(100) :
        if verbose :
            print(f'{k+1}/100')
        model = train_XGBoost(n_estimators,lr_min + (k/100)*(lr_max-lr_min),X_train,y_train)
        MAE.append(mae_model_preds(model,X_val,y_val))
    x = np.linspace(lr_min,lr_max,100)
    if display :
        plt.plot(x,MAE)
        plt.xlabel('learning rate')
        plt.ylabel('Mean Absolute Error')
    return (np.argmin(MAE)/100)*(lr_max-lr_min) + lr_min

#Best Mean Absolute Error with probabilities ouput is obtained with lr = 0.03
#Best Mean Absolute Error with prediction output is obtained with lr = 0.04258
# %%
