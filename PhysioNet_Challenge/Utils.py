##################################################################################################################################################################
#
# Libraries and global parameters
#
##################################################################################################################################################################

import os
import numpy as np
import pandas as pd
import scipy.stats
import librosa

root = 'data\\training_data'








##################################################################################################################################################################
#
# Preprocess
#
##################################################################################################################################################################

def get_locations(patientID,dataframe) :
    """
    Search patient information in the dataframe and returns a list of string with the recording location available
    If multiple recordings were given for a single location, select the first one and returns a modified to name to simplify the audio file research

    Args:
        patientID (int): Patient ID
        dataframe (Datframe): Dataframe

    Returns:
        array : array with the names of the location recordings audio files
    """    
    recording_locations = dataframe.loc[dataframe['Patient ID']==patientID].sample()['Recording locations:'].values[0].split('+')
    #Take one recording maximum per location
    uniques = np.unique(recording_locations).astype('<U10')
    #If there are multiple recording for a single location, take the first one and add "_1" to the audio file name
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
    """
    Search audio file in the data folder using the patient ID and the required recording location, and load it with librosa library

    Args:
        patientID (int): Patient ID
        location (str): Recording location

    Returns:
        array, int : sampled signal, sampling rate
    """    
    audio_file = str(patientID)+'_'+location+'.wav'
    filename = os.path.join(root,audio_file)
    recording, sampling_rate = librosa.load(filename)
    return recording, sampling_rate

    
    
def get_mfccs_features_location(patientID,location,n_mfcc) :
    """
    For a specified location, load the audio file and extract the mfcc feature

    Args:
        patientID (int): Patient ID
        location (str): Recording location
        n_mfcc (int):number of mel coefficients to extract

    Returns:
        array : 1D vector of mel coefficient features
    """    
    recording, sampling_rate = load_wavfile(patientID,location)
    mfccs = np.mean(librosa.feature.mfcc(y=recording,sr=sampling_rate,n_mfcc=n_mfcc).T,axis=0)
    features = np.array(mfccs).reshape([-1,1])
    return features


def get_mfccs_features(patientID,dataframe, n_mfcc) :
    """
    For a specified patient, extract the mfcc features from all available location

    Args:
        patientID (int): Patient ID
        dataframe (Datframe): Dataframe
        n_mfcc (int):number of mel coefficients to extract

    Returns:
        array : 1D vector (Flattened 2D array) containing the mfcc features of all location. Unavailable location are replaced with NaN
    """    
    locations = get_locations(patientID,dataframe)
    features = np.full([n_mfcc, 4], np.nan, dtype=float)
    location_order={'AV':0,'PV':1,'TV':2,'MV':3,'AV_1':0,'PV_1':1,'TV_1':2,'MV_1':3}
    for location in locations :
        if 'Phc' in location :
            break
        check = str(patientID)+'_'+location+'.wav'
        if os.path.isfile(os.path.join(root,check)) :
            features[:,location_order[location]] = get_mfccs_features_location(patientID,location,n_mfcc).reshape(n_mfcc,)
    return features.T.flatten()


def get_patient_features(patientID,dataframe) :
    """
    Extract the patient information from the Dataframe

    Args:
        patientID (int): Patient ID
        dataframe (Datframe): Dataframe

    Returns:
        array : 1D vector of patient information feature
    """    
    row = dataframe.loc[dataframe['Patient ID']==patientID].sample().drop(['Patient ID','Recording locations:','Sex','Pregnancy status','Murmur'],axis=1)
    values = row.values.astype('float32')
    features = values[0]
    return features


def get_signal_features(patientID,dataframe) :
    """
    For a specified patient, extract the global signal features (mean, variance and skew) from all available location

    Args:
        patientID (int): Patient ID
        dataframe (Datframe): Dataframe

    Returns:
        array : 1D vector (Flattened 2D array) containing the global signal features of all location. Unavailable location are replaced with NaN
    """    
    locations = get_locations(patientID,dataframe)
    features = np.full([3, 4], np.nan, dtype=float)
    location_order={'AV':0,'PV':1,'TV':2,'MV':3,'AV_1':0,'PV_1':1,'TV_1':2,'MV_1':3}
    for location in locations :
        if 'Phc' in location :
            break
        check = str(patientID)+'_'+location+'.wav'
        if os.path.isfile(os.path.join(root,check)) :
            recording, sampling_rate = load_wavfile(patientID,location)
            features[0,location_order[location]] = np.mean(recording)
            features[1,location_order[location]] = np.var(recording)
            features[2,location_order[location]] = scipy.stats.skew(recording)
    return features.T.flatten()


def get_features(patientID,dataframe,n_mfcc) :
    """
    Extract all features for a specified patient, using the previous functions, and concatenate into one array of ordered features

    Args:
        patientID (int): Patient ID
        dataframe (Datframe): Dataframe
        n_mfcc (int):number of mel coefficients to extract

    Returns:
        array : 1D vector with all ordered features of the patient
    """    
    mfccs_features = get_mfccs_features(patientID,dataframe,n_mfcc)
    patient_features = get_patient_features(patientID,dataframe)
    signal_features = get_signal_features(patientID,dataframe)
    features = np.concatenate((patient_features,mfccs_features,signal_features),axis=0)
    return features









##################################################################################################################################################################
#
# Main functions
#
##################################################################################################################################################################

def get_dataframe() :
    """
    Initialize the dataframe with selected useful columns and encoded variables

    Returns:
        Dataframe : Dataframe
    """    
    table = pd.read_csv('data\\training_data.csv')
    table = table[['Patient ID','Recording locations:','Age','Sex','Height','Weight','Pregnancy status','Murmur']]
    #only consider diagnosed patients
    table.drop(table[table.Murmur == 'Unknown'].index, inplace=True)
    
    #each Category is replaced with the number of months of its typical age, and categorical variables are encoded
    encoding={
        'Child':6*12,
        'Adolescent':15*12,
        'Infant':6,
        'Neonate':0.5,
        'Female':1,
        'Male':0,
        False:0,
        True:1
    }
    table['Age'] = table['Age'].map(encoding)
    table['Sex'] = table['Sex'].map(encoding)
    table['Pregnancy status'] = table['Pregnancy status'].map(encoding)
    return table


def get_X(dataframe,n_mfcc,verbose=1) :
    """
    Preprocess the data with the feature extraction defined above

    Args:
        dataframe (Datframe): Dataframe
        n_mfcc (int):number of mel coefficients to extract
        verbose (int, optional): Set to 0, 1 or 2 according to the desired level of precision when printing the progress of the process . Defaults to 1.

    Returns:
        array : 2D array with all patients features
    """    
    X = []
    if verbose >= 1 :
            print("Preprocessing X data...")
    for i in range(len(dataframe)) :
        if verbose >= 2 :
            print(f"Files processed :    {i+1}/{len(dataframe)}")
        patientID = dataframe.iloc[i]['Patient ID']
        current_features = get_features(patientID,dataframe,n_mfcc)
        X.append(current_features)
    if verbose >= 1 :
        print("Done.")
    return np.array(X)


def get_y(dataframe,verbose=1) :
    """
    Labels of the data

    Args:
        dataframe (Datframe): Dataframe
        verbose (int, optional): Set to 0, 1 or 2 according to the desired level of precision when printing the progress of the process . Defaults to 1.

    Returns:
        array : 1D array of labels 1.0 or 0.0
    """    
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
    """
    Display a signal example to visualize its shape

    Args:
        patientID (int): Patient ID
        location (str): Recording location
    """    
    import librosa.display
    recording, sampling_rate =  load_wavfile(patientID,location)
    librosa.display.waveshow(recording, sr=sampling_rate)
    

def determine_outcome(y_predict) :
    """
    Determine 1.0 or 0.0 predictions from the labels probabilities predicted by the model
    Adapt automatically to the output format of the 3 models

    Args:
        y_predict (array): Output of the model

    Returns:
        array : array of label predictions based on the output
    """    
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
    """
    Compute the representation of each class in the dataset, to evaluate imbalance

    Args:
        dataframe (Dataframe): Dataframe
    """    
    values = dataframe['Murmur'].value_counts().values
    absent = values[0]
    present = values[1]
    total = absent + present
    absent_ratio = round(10000*absent/total)/100
    present_ratio = round(10000*present/total)/100
    print(f'Absent :  {absent}  {absent_ratio}%')
    print(f'Present :  {present}    {present_ratio}%')