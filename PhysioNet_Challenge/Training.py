#%%
################################################################################
#
# Libraries, utils functions and global parameters
#
################################################################################

from Utils import *
import numpy as np
import matplotlib.pyplot as plt
import gc

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from sklearn.metrics import recall_score, precision_score, f1_score




data = get_dataframe()
root = 'data\\training_data'
n_mfcc = 10
random_state = 0

print('\nData intialized.\n')
print('Dataset balance :')
evaluate_dataset_balance(data)







#%%
################################################################################
#
# Preprocess
#
################################################################################

X = get_X(data,n_mfcc=n_mfcc)
y = get_y(data)

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, random_state=random_state, shuffle=True)

del X
del y
gc.collect()



imputer = SimpleImputer(missing_values=np.nan,strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train_upsample, y_train_upsample = SMOTE(random_state=random_state).fit_resample(X_train,y_train)


print('\nPreprocess completed\n')
print(f'X_train shape : {X_train.shape}')
print(f'y_train shape : {y_train.shape}')

print(f'X_test shape : {X_test.shape}')
print(f'y_test shape : {y_test.shape}\n')
print(f'Murmur ratio in the test sample : {y_test.mean()*100}%')




#%%
################################################################################
#
# Performance functions
#
################################################################################

def weighted_accuracy_score(y,preds) :
    y = pd.Series(y.flatten())
    preds = pd.Series(preds.flatten())
    TP = ((y==1)*(preds==1)).sum()
    TN = ((y==0)*(preds==0)).sum()
    FP = ((y==0)*(preds==1)).sum()
    FN = ((y==1)*(preds==0)).sum()
    score = (5*TP + TN)/(5*(TP+FN) + (TN+FP))
    return score


def performance(model,X_val,y_val,display=False) :
    output = model.predict(X_val)
    if len(output.shape) == 1 :
        output.reshape([-1,1])
    predictions = determine_outcome(output)
    weighted_accuracy = weighted_accuracy_score(y_val,predictions)
    precision = precision_score(y_val,predictions)
    recall = recall_score(y_val,predictions)
    f1 = f1_score(y_val,predictions)
    if display :
        print(f'Weighted Accuracy : {weighted_accuracy}')
        print(f'F1 : {f1}')
        print(f'Recall : {recall}')
        print(f'Precision : {precision}')
    return  weighted_accuracy, f1, recall, precision,



def score_model(model, params, display=False, cv=None) :
    if cv is None:
        cv = KFold(n_splits=5, random_state=random_state, shuffle=True)

    smoter = SMOTE(random_state=random_state)
    
    scores = []

    for train_fold_index, val_fold_index in cv.split(X_train, y_train):
        # Get the training data
        X_train_fold, y_train_fold = X_train[train_fold_index], y_train[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X_train[val_fold_index], y_train[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
        # Fit the model on the upsampled training data
        model_obj = model(**params).fit(X_train_fold_upsample, y_train_fold_upsample)
        # Score the model on the (non-upsampled) validation data
        score = performance(model_obj,X_val_fold,y_val_fold)
        scores.append(score)
    scores = np.array(scores)
    performances=[np.nan for j in range(4)]
    for j in range(4) :
        performances[j] = np.mean(scores[:,j])
    [weighted_accuracy, f1, recall, precision] = performances
    if display :
        print(f'Weighted Accuracy : {weighted_accuracy}')
        print(f'F1 : {f1}')
        print(f'Recall : {recall}')
        print(f'Precision : {precision}')
    return  weighted_accuracy, f1, recall, precision







    

# %%
################################################################################
#
# Linear Regression model
#
################################################################################

parameters = {}
model_LR = LinearRegression(**parameters)
model_LR.fit(X_train_upsample,y_train_upsample)
performance(model_LR, X_test, y_test, True)




# %%
################################################################################
#
# Random Forest model
#
################################################################################

def select_best_parameters_RF(parameters, display=True) :
    score_tracker = []
    for n_estimators in parameters['n_estimators'] :
        for max_depth in parameters['max_depth'] :
            example_parameters = {
                'n_estimators' : n_estimators,
                'max_depth' : max_depth,
                'random_state' : random_state
            }
            example_parameters['Weighted Accuracy'] = score_model(RandomForestClassifier,example_parameters)[0]
            score_tracker.append(example_parameters)
        
    best_parameters = sorted(score_tracker, key= lambda x: x['Weighted Accuracy'], reverse=True)[0]
    best_parameters.popitem()
    best_parameters.popitem()
    if display :
        print(f"Best parameters are : {best_parameters}")
    return best_parameters


parameters = {
    'n_estimators' : [50,75,100,125,150],
    'max_depth' : [3,5,7],
    'random_state' : [random_state]
}
best_parameters_RF = select_best_parameters_RF(parameters)

#%%
model_RF = RandomForestClassifier(**best_parameters_RF)
model_RF.fit(X_train_upsample,y_train_upsample)

print('\n')
performance(model_RF, X_test, y_test, True)




# %%
################################################################################
#
# XGBoost model
#
################################################################################

def select_best_parameters_XGB(parameters, display = True) :
    score_tracker = []
    for n_estimators in parameters['n_estimators'] :
        for learning_rate in parameters['learning_rate'] :
            example_parameters = {
                'n_estimators' : n_estimators,
                'learning_rate' : learning_rate,
                'random_state' : random_state
            }
            example_parameters['Weighted Accuracy'] = score_model(XGBRegressor,example_parameters)[0]
            score_tracker.append(example_parameters)
        
    best_parameters = sorted(score_tracker, key= lambda x: x['Weighted Accuracy'], reverse=True)[0]
    best_parameters.popitem()
    best_parameters.popitem()
    if display :
        print(f"Best parameters are : {best_parameters}")
    return best_parameters


parameters = {
    'n_estimators' : [750,1000,1500],
    'learning_rate' : [0.00001,0.0001,0.001],
    'random_state' : [random_state]
}
best_parameters_XGB = select_best_parameters_XGB(parameters)

#%%
model_XGB = XGBRegressor(**best_parameters_XGB)
model_XGB.fit(X_train_upsample,y_train_upsample)

print('\n')
performance(model_XGB, X_test, y_test, True)











#%%
################################################################################
#
# Feature importance
#
################################################################################

def feature_importance(model) :
    patient_feature_names = ["Age","Height","Weight"]
    blank = ["" for i in range(n_mfcc -1)]
    mfccs_feature_names = ["mfccs features AV"] + blank + ["mfccs features PV"] + blank + ["mfccs features TV"] + blank + ["mfccs features MV"] + blank
    blank = ["",""]
    signal_feature_names = ["signal features AV"] + blank + ["signal features PV"] + blank + ["signal features TV"] + blank + ["signal features MV"] + blank
    feature_names = patient_feature_names + mfccs_feature_names+signal_feature_names
    
    importances = model.feature_importances_
    importances = pd.Series(importances, index=feature_names)
    
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return importances

    
feature_importance(model_XGB)

# %%
