# %%
from Challenge_discovery.Load_data import *
import pandas as pd
import scipy.stats



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

        # Extract labels
        current_murmur = get_murmur(current_patient_data)
        murmurs.append(current_murmur)
        
    if verbose >= 1:
        print('Done.')
    return features,murmurs





# %%
features, murmurs = extract_features_and_labels('data\\training_data', 0)


# %%
def repartition_labels(murmurs) :
    count = [0,0,0]
    onehot = {'Present':0,'Unknown':1,'Absent':2}
    for label in murmurs :
        count[onehot[label]] += 1
    stats = {'Present':count[0],'Unknown':count[1],'Absent':count[2]}
    return stats
repartition_labels(murmurs)
# %%
murmurs = pd.DataFrame(murmurs)
murmurs.columns=['label']
murmurs
count = murmurs.groupby('label').value_counts()
count.plot(kind='pie', subplots=True, figsize=(8, 8))
# %%
