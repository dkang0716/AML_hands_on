
import argparse
import os
import dill
from azureml.core import Run
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from utils import Prep, Modelling

# let user feed in 1 parameter, the location of the data files (from datastore)
parser = argparse.ArgumentParser()
parser.add_argument('--dataFolderInput', type=str)
parser.add_argument('--dataFolderOutput', type=str)

args = parser.parse_args()
dataFolderInput = args.dataFolderInput
dataFolderOutput = args.dataFolderOutput
 
# load train set into numpy arrays
df = pd.read_csv(os.path.join(dataFolderInput, 'rti_14actions/rti_14actions.csv'),sep=";", index_col=0)

dict_model = {}

for i in df.columns[1:]:
    
    serie = pd.Series(df[i].values, index=pd.to_datetime(df.iloc[:,0]) )

    p1 = Prep(step_len=2)

    y_train, X_train = p1.fit_transform(serie)

    #print(X_train.shape, y_train.shape, sep = '\n')

    p2 = Modelling(model = LinearRegression())
    
    p2.fit(X_train, y_train)
    
    dict_model.update({i: [p1,p2]}) 

os.makedirs(dataFolderOutput, exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
filename=os.path.join(dataFolderOutput,'rti_14actions/sklearn_rti_14models.pkl')
dill.dump(dict_model,open(filename,'wb'))
