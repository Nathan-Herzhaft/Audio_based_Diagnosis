# Audio-based Diagnosis

## General Description
This folder contains a commented pipeline of an audio based diagnosis using machine learning. This algorithm was designed as part of the PhysioNet Challenge 2022, the goal is to diagnose heart murmur using the analysis of a phonocardiogram.  
The folder is provided with a *Diagnosis.py* file, which is the main file for training and evaluation of the models. It relies on a *Utils.py* file containing all the utils function, stored in a separate file for a better readability.  
Link of the Challenge : https://moody-challenge.physionet.org/2022/  

---

## 1. Libraries  

We use several libraries that need to be installed upstream :
- Librosa
- Sklearn
- Pandas
- Matplotlib
- Numpy

### > Librosa
Librosa is a python library for audio and music processing. We use it to load wavefiles and process them with mel coefficient extraction  
Link : https://librosa.org/

---

## 2. Trainig data  

The data we use for training is provided by the challenge in opensource. It was collected from a pediatric population during two mass screening campaigns. Each patient in the data has one or more recordings from one or mor auscultation location : pulmonary valve (PV), aortic valve (AV), mitral valve (MV), and tricuspid valve (TV). The number, location, and duration of the recordings vary between patients.  
The data needs to be downloaded from the Challenge website and stored in a floder named *data* to be read automatically by the python script.

---

## 3. Main python file description
The *Diagnosis* file is divided into 7 segments which can be compiled separately for a better understanding, and simplify script modifications.

1. Libraries, utils functions and global parameters
2. Preprocess
3. Performance measures
4. Linear Regression model
5. Random Forest model
6. XGBoost model
7. Feature importance

## > Libraries, utils functions and global parameters
First segment is simply the initialization of ressources and parameters. After importing libraries, we link the main file to the *Utils* file to import the utils functions, and define global parameters such as the data set or the folder root.  

## > Preprocess
Relying on *Utils* file, we execute the preprocess pipeline and store the data in X and y variables. After a train/test split, we use a simple Imputer for missing values and perform data augmentation using SMOTE method to get a balanced dataset for training.

## > Performance measures
Third segment is dedicated to the definition of the functions used for the performances measurement. The weighted_accuracy is the metric used by the chaleenge to sort the competitors, and it is the one we'll use to optimize our model parameters. Other metrics score are also computed to get a better comprehension of the model performances.

## > Linear Regression model
The first model we study is a simple Linear Regression. It doesn't need parameters tuning, so we can directly define our model and train it on the data-augmented training set.

## > Random Forest model
The second model is Random Forest. We define a function to optimize the parameters. The function is easily editable in case you need to change the parameters to optimize. After the tuning, we can define a model according to the optimized parameters and train it on the data-augmented training set.

## > XGBoost model
Finally, we study XGBoost model. Similarly to Random Forest, we first optimize parameters then train the model with data-augmented training set.

## > Feature importance
After the different model comparison, we can display the feature importance according to each model to understand which feature are relevant for a heart murmur diagnostic.

---

## 4. Utils python file description
The *Utils* file stores the different functions to allow a better readability of the main file.  
  
First functions presented are preprocessing functions. To preprocess the data, we load audio file for each patient in the dataset and extract the *Mel Frequency Cepstrum Coefficient* or *MFCC* of the signal. These coefficients are used to describe an audio signal and we use them as features for our machine learning models. On top of these *MFCC*, we use as features the patient information such as Height, Weight, Age, as well as more general features of the audio signal such as mean variance and skew.  
  
After the preprocess, we define functions that are callable directly by the main file. *get_dataframe* to intitalize the dataframe, *get_X* and *get_y* to load the preprocess data, and other useful functions for specific tasks of the main script.