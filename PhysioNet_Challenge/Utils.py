#%%
################################################################################
#
# Libraries and global parameters
#
################################################################################

import os
import numpy as np
import pandas as pd
import glob
import librosa

root = 'data\\training_data'
n_mfcc = 20
random_state = 0








################################################################################
#
# Preprocess features
#
################################################################################

def get_locations(patientID,dataframe) :
    recording_locations = dataframe.loc[patientID]['Recording locations:'].split('+')
    uniques = np.unique(recording_locations).astype('<U10')
    if len(set(recording_locations)) != len(recording_locations) :
        seen = {}
        for location in recording_locations :
            if location not in seen :
                seen[location] = 1
            else :
                if seen[location] == 1 :
                    new_text = uniques[np.where(uniques==location)[0][0]] + '_1'
                    uniques[np.where(uniques==location)] = new_text
                    seen[location] += 1
    return uniques


def load_wavfile(patientID,location) :
    audio_file = str(patientID)+'_'+location+'.wav'
    filename = os.path.join(root,audio_file)
    recording, sampling_rate = librosa.load(filename)
    return recording, sampling_rate

    
    
def get_mfccs_features_location(patientID,location) :
    recording, sampling_rate = load_wavfile(patientID,location)
    mfccs = np.mean(librosa.feature.mfcc(y=recording,sr=sampling_rate,n_mfcc=n_mfcc).T,axis=0)
    features = np.array(mfccs).reshape([-1,1])
    return features


def get_mfccs_features(patientID,dataframe) :
    locations = get_locations(patientID,dataframe)
    features = np.zeros((n_mfcc, 5), dtype=float)
    location_order={'AV':0,'PV':1,'TV':2,'MV':3,'Phc':4,'AV_1':0,'PV_1':1,'TV_1':2,'MV_1':3,'Phc_1':4}
    for location in locations :
        check = str(patientID)+'_'+location+'.wav'
        if os.path.isfile(os.path.join(root,check)) :
            features[:,location_order[location]] = get_mfccs_features_location(patientID,location).reshape(20,)
    return features.T.flatten()


def get_patient_features(patientID,dataframe) :
    row = dataframe.loc[patientID].drop(['Recording locations:','Murmur'])
    values = row.values.astype('float32')
    features = values
    return features


def get_features(patientID,dataframe) :
    mfccs_features = get_mfccs_features(patientID,dataframe)
    patient_features = get_patient_features(patientID,dataframe)
    features = np.concatenate((patient_features,mfccs_features),axis=0)
    return features









################################################################################
#
# Utils functions
#
################################################################################

def get_dataframe() :
    table = pd.read_csv('data\\training_data.csv')
    table = table[['Patient ID','Recording locations:','Age','Sex','Height','Weight','Pregnancy status','Murmur']]
    table.drop(table[table.Murmur == 'Unknown'].index, inplace=True)
    table.set_index('Patient ID', inplace=True)
    
    onehot={
        'Child':6*12,
        'Adolescent':15*12,
        'Infant':6,
        'Neonate':0.5,
        'Female':1,
        'Male':0,
        False:0,
        True:1
    }
    table['Age'] = table['Age'].map(onehot)
    table['Sex'] = table['Sex'].map(onehot)
    table['Pregnancy status'] = table['Pregnancy status'].map(onehot)
    return table


def get_X(dataframe,verbose=1) :
    X = []
    if verbose >= 1 :
            print("Preprocessing X data...")
    for i in range(len(dataframe)) :
        if verbose >= 2 :
            print(f"Files processed :    {i+1}/{len(dataframe)}")
        patientID = dataframe.iloc[i].name
        current_features = get_features(patientID,dataframe)
        X.append(current_features)
    if verbose >= 1 :
        print("Done.")
    return np.array(X)


def get_y(dataframe,verbose=1) :
    y=[]
    onehot = {
        'Absent':0.0,
        'Present':1.0
    }
    if verbose >= 1 :
        print("Preprocessing y data...")
    for i in range(len(dataframe)) :
        if verbose >= 2 :
            print(f"Files processed :    {i+1}/{len(dataframe)}")
        y.append(onehot[dataframe.iloc[i]['Murmur']])
    if verbose >= 1 :
        print("Done.")
    return np.array(y).reshape([-1,1])


def display_signal(patientID,location) :
    import librosa.display
    recording, sampling_rate =  load_wavfile(patientID,location)
    librosa.display.waveshow(recording, sr=sampling_rate)
    

def determine_outcome(y_predict) :
    for i in range(len(y_predict)) :
        if type(y_predict[i]) == np.ndarray :
            if y_predict[i][0] > 0.5 :
                y_predict[i][0] = 1
            else :
                y_predict[i][0] = 0
        else :
            if y_predict[i] > 0.5 :
                y_predict[i] = 1
            else :
                y_predict[i] = 0
    return y_predict


def evaluate_dataset_balance(dataframe) :
    values = dataframe['Murmur'].value_counts().values
    absent = values[0]
    present = values[1]
    total = absent + present
    absent_ratio = round(10000*absent/total)/100
    present_ratio = round(10000*present/total)/100
    print(f'Normal ratio :  {absent_ratio} %')
    print(f'Abnormal ratio :  {present_ratio} %')
    
    
#%%
import sklearn
data = get_dataframe()
recording, sampling_rate = load_wavfile(9979,'AV')
mfccs = librosa.feature.mfcc(recording, sr=sampling_rate)
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
# %%
