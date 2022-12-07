# Data Challenge INF554

## General Description
This jupyter notebook contains a commented pipeline to predict a number of retweet using machine learning.
Our final model uses a Word2vec algorith and an XGBoost classifier. We describe in our report the choices that led to this model.

---

## 1. Libraries  

We use several libraries that need to be installed upstream :
- Numpy
- Pandas
- Matplotlib
- sklearn
- xgboost
- gensim

### > XGBoost
XGBoost is a python library that implements an algorithm of gradient boosting. We use it to as a classifier and compare it to other regression methods.  
Link : https://xgboost.readthedocs.io/en/stable/

---

## 2. Training data  

The data we use for training is provided by the challenge in opensource.
The data can be used directly from kaggle on the dataset page, or using an API. If you choose to use it locally, you simply have to change the path of the file in the functions pd.read_csv() 

---

## 3. Main python file description
The notebook is divided into few segments which can be compiled separately for a better understanding, and simplify script modifications.

1. Libraries, data loading
2. Preprocess using word embedding
3. Models Comparison
4. Feature importance
5. Final model and predictions
6. Write the csv submission

## > Libraries, data loading
First segment is simply the initialization of ressources and parameters. After importing libraries, we load our data and modify the index of our dataframes.

## > Preprocess using word embedding
Using Word2vec, we set up 5 embedding features and add them to our training dataset (after we trained it on the vocabulary available). On top of that, we drop the columns not required by the regression to simply the model.

## > Models Comparison
For each considered model, we fine the optimum parameters using a grid search (except for linear regression, that doesn't need it). Then, we can find the performing model that we will use to make our predictions.

## > Final model and predictions
Once we chose our model, we can train it on all the training_data available (we do not need a train test split anymore), and make our predictions. After this, we can refine briefly the predictions to get only natural integers, as required for a number of retweets.

## > Feature importance
After we found the finest model, it can be nice to visualize the features that were the most relevant to make the predictions. This can help us understand how we should modify our data to make better results (cf. the Timestamp modifications in the report)

## > Write the csv submission
Finally, we can write our predictions in a csv file that is in the right format for the competition 
  
---

Nathan Herzhaft : nathan.herzhaft@polytechnique.edu
Tom Kasprzak : tom.kasprzak@polytechnique.edu