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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.metrics import accuracy_score


root = 'data\\training_data'
n_mfcc = 20
random_state = 0
data = get_dataframe().sample(100,random_state=random_state)









#%%
################################################################################
#
# Preprocess
#
################################################################################

X = get_X(data)
y = get_y(data)

imputer = SimpleImputer().fit(X)
X = imputer.transform(X)
X_train,X_val,y_train,y_val = train_test_split(X, y, train_size=0.8, random_state=random_state, shuffle=True)


del X
del y
gc.collect()


print('\nPreprocess completed\n')
print(f'X_train shape : {X_train.shape}')
print(f'y_train shape : {y_train.shape}')
print(f'X_val shape : {X_val.shape}')
print(f'y_val shape : {y_val.shape}\n')

print('Dataset balance :')
evaluate_dataset_balance(data)






#%%
################################################################################
#
# Evaluation functions
#
################################################################################

def evaluate_accuracy(model,X_val,y_val) :
    predictions = model.predict(X_val)
    if len(predictions.shape) == 1 :
        predictions.reshape([-1,1])
    output = determine_outcome(predictions)
    return accuracy_score(output,y_val)


def evaluate_mean_absolute_error(model,X_val,y_val) :
    output = model.predict(X_val)
    return mean_absolute_error(y_val,output)


def evaluate_precision(model,X_val,y_val) :
    predictions = model.predict(X_val)
    if len(predictions.shape) == 1 :
        predictions.reshape([-1,1])
    output = determine_outcome(predictions)
    return precision_score(y_val,output)


def evaluate_recall(model,X_val,y_val) :
    predictions = model.predict(X_val)
    if len(predictions.shape) == 1 :
        predictions.reshape([-1,1])
    output = determine_outcome(predictions)
    return recall_score(y_val,output)


def performance_model(model) :
    mean_absolute_error = evaluate_mean_absolute_error(model,X_val,y_val)
    accuracy = evaluate_accuracy(model,X_val,y_val)
    precision = evaluate_precision(model,X_val,y_val)
    recall = evaluate_recall(model,X_val,y_val)
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Mean Absolute Error : {mean_absolute_error}')
    print(f'Accuracy : {accuracy}')
    return precision, recall, mean_absolute_error, accuracy








    

# %%
################################################################################
#
# Linear Regression model
#
################################################################################

def linear_regression() :
    model = LinearRegression(
        fit_intercept = True, normalize = True, copy_X = True
    )
    return model

model = linear_regression()
model.fit(X_train,y_train)
perf = performance_model(model)







# %%
################################################################################
#
# Random Forest model
#
################################################################################

def random_forest(n_estimators,max_depth,max_features) :
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        max_features = max_features,
        random_state = random_state
        
    )
    return model

model = random_forest(1000,7,110)
model.fit(X_train,y_train)
perf = performance_model(model)



# %%
