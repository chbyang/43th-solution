# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:11:51 2018

@author: aa
"""
import numpy as np
import pandas as pd 

lgb = pd.read_csv('./180715 lgbm_0.799.csv')
#xgb = pd.read_csv('./fold11_xgb_score_0.796.csv')
#lgb2 = pd.read_csv('./bureau_lgbm_0.793.csv')
lgb3= pd.read_csv('./lgb_cv0.79765.csv')
blend1 =pd.read_csv('./shaz13_0.8.csv')
blend2 =pd.read_csv('./Ishaan_0.8.csv')
xgb = pd.read_csv('./xgb_raw_predicton_0.797.csv')

lgb['TARGET']=xgb['TARGET']*0.45+lgb3['TARGET']*0.55

lgb.to_csv('sub.csv',index=False)