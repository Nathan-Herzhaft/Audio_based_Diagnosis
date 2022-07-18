#%%
from Load_data import *
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %%
def train_challenge_model(data_folder, verbose):
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

    X = np.vstack(features)
    y = np.vstack(murmurs)
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 45   # Maximum number of leaf nodes in each tree.
    random_state   = 6789 # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(X_train)
    y_train = imputer.transform(y_train)
    murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(X_train, y_train)

    if verbose >= 1:
        print('Done.')
        
    return murmur_classifier




# %%
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


def train_Random_Forest(n_estimators,max_leaf_nodes,X_train, y_train) :
    model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes)
    model.fit(X_train,y_train)
    return model
    
    
def test_model(model, X_val, y_val,display=False) :
    predictions = model.predict(X_val)
    MAE = mean_absolute_error(predictions, y_val)
    if display :
        print("Mean Absolute Error: " + str(MAE))
    return MAE
    
    
# %%
features,murmurs = extract_features_and_labels('data\\training_data', 1)
X_train, X_val, y_train, y_val = generate_input(features,murmurs)



# %%
def compare_efficiency_estimators(n_min,n_max,max_leaf_nodes,X_train, X_val, y_train, y_val, display=True, verbose = False) :
    MAE = []
    for n in range(n_min,n_max) :
        if verbose :
            print(f'{n-n_min+1}/{n_max-n_min}')
        model = train_Random_Forest(n,max_leaf_nodes,X_train,y_train)
        MAE.append(test_model(model,X_val,y_val))
    x = np.linspace(n_min,n_max,1)
    if display :
        plt.plot(x,MAE)
    return MAE
        
        
# %%
compare_efficiency_estimators(50,500,10,X_train,X_val,y_train,y_val,True,True)
# %%
