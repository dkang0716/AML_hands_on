import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from xgboost import XGBRegressor
#import matplotlib.pyplot as plt
#from jours_feries_france.compute import JoursFeries

class Prep(BaseEstimator, TransformerMixin):
    def __init__(self, fillna='pad',step_forecast=1, step_len=7):
        self.fillna = fillna
        self.step_forecast = step_forecast
        self.step_len = step_len
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        #### fill na###################
        X = X.fillna(method=self.fillna).dropna()
        
        df,df_name =[], []
        
        #### delay ###################
        for i in range(self.step_len):
            df.append(X.shift(self.step_forecast + i))
            df_name.append('step_delay_{}'.format(i))
            
            
        ### day of week#################
        df.append(pd.Series(X.index.dayofweek, index=X.index))
        df_name.append('day_of_week')
        
       ### jour férié france##############
#        if X.index.year.nunique()>=2:
#             list_jf = []
#             for i in range(min(X.index.year),max(X.index.year)):
#                 list_jf.append(list(JoursFeries.for_year(i).values()))
#             list_jf_years = np.concatenate(list_jf)
#         else:
#             list_jf_years = list(JoursFeries.for_year(np.unique(X.index.year)).values())
       
#         df.append(pd.Series(X.index.isin(list_jf_years), index=X.index))
#         df_name.append('jour_ferie')
        
        
        df=pd.concat(df, axis=1)
        df.columns=df_name
        
        ###moving diff################
        df['diff_mov'] = df['step_delay_0']-df['step_delay_1']
        
        return X[(self.step_len+self.step_forecast-1):],df[(self.step_len+self.step_forecast-1):]
            
    
class Modelling(BaseEstimator, RegressorMixin):
    
    def __init__(self, model=XGBRegressor(), **param):
        self.param = param
        self.model = model
    
    def fit(self, X, y):
        self.model.set_params(**self.param)
        self.model.fit(X, y)
        return self
    
    def predict(self, X, y=None):
        return pd.Series(self.model.predict(X), index = X.index)
    
    def score(self, X, y):
        p = self.predict(X)
        mae, rmse = abs(p - y).mean(), np.sqrt(np.power((p - y),2).mean())
        return mae, rmse
    
    def plot_obs_pred(self, X, y):
        p = self.predict(X)
        pd.DataFrame({'obs':y, 'pred':p}).plot()
        plt.show()
