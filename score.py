
import numpy as np
import os
import dill
import joblib
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from utils import Prep, Modelling

from azureml.core.model import Model

# let user feed in 1 parameter, the location of the data files (from datastore)
parser = argparse.ArgumentParser()
parser.add_argument('--dataFolderInput', type=str)
parser.add_argument('--modelFolderInput', type=str)
parser.add_argument('--dataFolderOutput', type=str)

args = parser.parse_args()
dataFolderInput = args.dataFolderInput
modelFolderInput = args.modelFolderInput
dataFolderOutput = args.dataFolderOutput

# load train set into numpy arrays
df = pd.read_csv(os.path.join(dataFolderInput, 'rti_14actions/rti_14actions.csv'),sep=";", index_col=0)

file_name = os.path.join(modelFolderInput,'rti_14actions/sklearn_rti_14models.pkl')
print(file_name)

dict_model = dill.load(open(file_name , 'rb'))

predicts = []
for i in df.columns[1:]:
    
    serie = pd.Series(df[i].values, index=pd.to_datetime(df.iloc[:,0]) )

    X_test = dict_model.get(i)[0].transform(serie)[1]
    
    predict = pd.DataFrame({'action':i, 'observation': dict_model.get(i)[0].transform(serie)[0]}, index=pd.to_datetime(df.iloc[:,0]))
    predict['prediction'] = dict_model.get(i)[1].predict(X_test)
    
    predicts.append(predict)
    
prediction = pd.concat(predicts)

pd.DataFrame(prediction).to_csv(os.path.join(dataFolderOutput, 'rti_14actions/Prediction_RTI_14actions.csv'))
