#https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# File system manangement
#import os
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Training data
#app_train = pd.read_csv('../input/application_train.csv')
app_train = pd.read_csv('./train_cleaned.csv')
print('Training data shape: ', app_train.shape)
#print(app_train.head())
# Testing data features
#app_test = pd.read_csv('../input/application_test.csv')
app_test = pd.read_csv('./test_cleaned.csv')
print('Testing data shape: ', app_test.shape)
#print(app_test.head())
# Number of each type of column
print(app_train.dtypes.value_counts())
# Number of unique classes in each object column
print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

docs = [_f for _f in app_train.columns if 'FLAG_DOC' in _f]
live = [_f for _f in app_train.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

app_train['NEW_DOC_IND_AVG'] = app_train[docs].mean(axis=1)
app_train['NEW_DOC_IND_STD'] = app_train[docs].std(axis=1)
app_train['NEW_DOC_IND_KURT'] = app_train[docs].kurtosis(axis=1)
app_train['NEW_LIVE_IND_SUM'] = app_train[live].sum(axis=1)
app_train['NEW_LIVE_IND_STD'] = app_train[live].std(axis=1)
app_train['NEW_LIVE_IND_KURT'] = app_train[live].kurtosis(axis=1)

app_test['NEW_DOC_IND_AVG'] = app_test[docs].mean(axis=1)
app_test['NEW_DOC_IND_STD'] = app_test[docs].std(axis=1)
app_test['NEW_DOC_IND_KURT'] = app_test[docs].kurtosis(axis=1)
app_test['NEW_LIVE_IND_SUM'] = app_test[live].sum(axis=1)
app_test['NEW_LIVE_IND_STD'] = app_test[live].std(axis=1)
app_test['NEW_LIVE_IND_KURT'] = app_test[live].kurtosis(axis=1)


app_train = pd.get_dummies(app_train, dummy_na=True)
app_test = pd.get_dummies(app_test, dummy_na=True)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
# Number of each type of column
print(app_train.dtypes.value_counts())
# Number of unique classes in each object column
print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

app_train=app_train.loc[:,(app_train!=0).any(axis=0)]
app_test=app_test.loc[:,(app_test!=0).any(axis=0)]


print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
# Number of each type of column
print(app_train.dtypes.value_counts())
# Number of unique classes in each object column
print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))


train_labels = app_train['TARGET']
# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
# Add the target back in
app_train['TARGET'] = train_labels
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

app_train['DAYS_BIRTH']=app_train['DAYS_BIRTH']/(-365)
app_train['DAYS_EMPLOYED']=app_train['DAYS_EMPLOYED']/(-365)
app_train['DAYS_REGISTRATION']=app_train['DAYS_REGISTRATION']/(-365)
app_train['DAYS_ID_PUBLISH']=app_train['DAYS_ID_PUBLISH']/(-365)
app_train['DAYS_LAST_PHONE_CHANGE']=app_train['DAYS_LAST_PHONE_CHANGE']/(-365)

app_test['DAYS_BIRTH']=app_test['DAYS_BIRTH']/(-365)
app_test['DAYS_EMPLOYED']=app_test['DAYS_EMPLOYED']/(-365)
app_test['DAYS_REGISTRATION']=app_test['DAYS_REGISTRATION']/(-365)
app_test['DAYS_ID_PUBLISH']=app_test['DAYS_ID_PUBLISH']/(-365)
app_test['DAYS_LAST_PHONE_CHANGE']=app_test['DAYS_LAST_PHONE_CHANGE']/(-365)

#app_train.to_csv('./dummies_train.csv',index=False)
#app_test.to_csv('./dummies_test.csv',index=False)



#apply domain knowledge
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()
app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / (1+app_train_domain['AMT_INCOME_TOTAL'])
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / (1+app_train_domain['AMT_INCOME_TOTAL'])
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['PRICE_INCOME_PERCENT'] = app_train_domain['AMT_GOODS_PRICE'] / (1+app_train_domain['AMT_INCOME_TOTAL'])
app_train_domain['CREDIT_PRICE_PERCENT'] = app_train_domain['AMT_CREDIT'] / (1+app_train_domain['AMT_GOODS_PRICE'])
app_train_domain['AMT_CREDIT - AMT_GOODS_PRICE'] = app_train_domain['AMT_CREDIT'] - app_train_domain['AMT_GOODS_PRICE']
app_train_domain['AMT_INCOME_TOTAL - AMT_ANNUITY'] = app_train_domain['AMT_INCOME_TOTAL'] - app_train_domain['AMT_ANNUITY']
app_train_domain['AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = app_train_domain['AMT_INCOME_TOTAL'] / 12. - app_train_domain['AMT_ANNUITY']
app_train_domain['AMT_INCOME_TOTAL - AMT_CREDIT'] = app_train_domain['AMT_INCOME_TOTAL'] - app_train_domain['AMT_CREDIT']
app_train_domain['AMT_INCOME_TOTAL / CREDIT_TERM - AMT_CREDIT'] = app_train_domain['AMT_INCOME_TOTAL']/app_train_domain['CREDIT_TERM'] - app_train_domain['AMT_CREDIT']
#app_train_domain['AMT_INCOME_TOTAL*5 - AMT_CREDIT'] = app_train_domain['AMT_INCOME_TOTAL']*5 - app_train_domain['AMT_CREDIT']
#app_train_domain['AMT_INCOME_TOTAL*10 - AMT_CREDIT'] = app_train_domain['AMT_INCOME_TOTAL']*10 - app_train_domain['AMT_CREDIT']
#app_train_domain['AMT_INCOME_TOTAL*20 - AMT_CREDIT'] = app_train_domain['AMT_INCOME_TOTAL']*20 - app_train_domain['AMT_CREDIT']
app_train_domain['AMT_GOODS_PRICE / AMT_ANNUITY'] = app_train_domain['AMT_GOODS_PRICE'] / app_train_domain['AMT_ANNUITY']
app_train_domain['AMT_GOODS_PRICE - AMT_ANNUITY'] = app_train_domain['AMT_GOODS_PRICE'] - app_train_domain['AMT_ANNUITY']
app_train_domain['AMT_GOODS_PRICE - AMT_INCOME_TOTAL'] = app_train_domain['AMT_GOODS_PRICE'] - app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['AMT_CREDIT - AMT_ANNUITY'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_ANNUITY']

app_train_domain['app DAYS_EMPLOYED - DAYS_BIRTH'] = app_train_domain['DAYS_EMPLOYED'] - app_train_domain['DAYS_BIRTH']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_train_domain['CAR_TO_BIRTH_RATIO_trick'] = (app_train_domain['OWN_CAR_AGE'] / app_train_domain['DAYS_BIRTH'])
app_train_domain['CAR_TO_BIRTH_RATIO'] = (app_train_domain['OWN_CAR_AGE'] / app_train_domain['DAYS_BIRTH'])*app_train_domain['FLAG_OWN_CAR_Y']
app_train_domain['PHONE_TO_BIRTH_RATIO'] = app_train_domain['DAYS_LAST_PHONE_CHANGE'] / app_train_domain['DAYS_BIRTH']
app_train_domain['CAR_TO_EMPLOY_RATIO_trick'] = (app_train_domain['OWN_CAR_AGE'] / app_train_domain['DAYS_EMPLOYED'])
app_train_domain['CAR_TO_EMPLOY_RATIO'] = (app_train_domain['OWN_CAR_AGE'] / app_train_domain['DAYS_EMPLOYED'])*app_train_domain['FLAG_OWN_CAR_Y']
app_train_domain['PHONE_TO_EMPLOY_RATIO'] = app_train_domain['DAYS_LAST_PHONE_CHANGE'] / app_train_domain['DAYS_EMPLOYED']


app_train_domain['THREE_SOURDES'] = app_train_domain['EXT_SOURCE_1'] * app_train_domain['EXT_SOURCE_2']* app_train_domain['EXT_SOURCE_3']
app_train_domain['app EXT_SOURCE std'] = app_train_domain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
app_train_domain['app EXT_SOURCE_1 * EXT_SOURCE_2'] = app_train_domain['EXT_SOURCE_1'] * app_train_domain['EXT_SOURCE_2']
app_train_domain['app EXT_SOURCE_1 * EXT_SOURCE_3'] = app_train_domain['EXT_SOURCE_1'] * app_train_domain['EXT_SOURCE_3']
app_train_domain['app EXT_SOURCE_2 * EXT_SOURCE_3'] = app_train_domain['EXT_SOURCE_2'] * app_train_domain['EXT_SOURCE_3']
app_train_domain['app EXT_SOURCE_1 * DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_1'] * app_train_domain['DAYS_BIRTH']
app_train_domain['app EXT_SOURCE_2 * DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_2'] * app_train_domain['DAYS_BIRTH']
app_train_domain['app EXT_SOURCE_3 * DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_3'] * app_train_domain['DAYS_BIRTH']
app_train_domain['app EXT_SOURCE_1 / DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_1'] / app_train_domain['DAYS_BIRTH']
app_train_domain['app EXT_SOURCE_2 / DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_2'] / app_train_domain['DAYS_BIRTH']
app_train_domain['app EXT_SOURCE_3 / DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_3'] / app_train_domain['DAYS_BIRTH']

app_train_domain['EXT_SOURCE_AVE / DAYS_BIRTH'] = app_train_domain['EXT_SOURCE_AVE'] / app_train_domain['DAYS_BIRTH']
app_train_domain['EXT_SOURCE_AVE / DAYS_REGISTRATION'] = app_train_domain['EXT_SOURCE_AVE'] / app_train_domain['DAYS_REGISTRATION']
app_train_domain['EXT_SOURCE_AVE / DAYS_ID_PUBLISH'] = app_train_domain['EXT_SOURCE_AVE'] / app_train_domain['DAYS_ID_PUBLISH']

app_train_domain['INCOME_PER_PERSON']=app_train_domain['AMT_INCOME_TOTAL']/(app_train_domain['CNT_FAM_MEMBERS'])
#app_train_domain['CHILDREN_TO_PARENTS_RATIO']=app_train_domain['CNT_CHILDREN']/(app_train_domain['CNT_PARENTS_MEMBERS'])
#app_train_domain['CHILDREN_TO_FAMILY_RATIO']=app_train_domain['CNT_CHILDREN']/(app_train_domain['CNT_FAM_MEMBERS'])
app_train_domain['INC_PER_CHLD']=app_train_domain['AMT_INCOME_TOTAL']/(1+app_train_domain['CNT_CHILDREN'])

  
app_train_domain['app most popular AMT_GOODS_PRICE'] = app_train_domain['AMT_GOODS_PRICE'].isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
app_train_domain['app popular AMT_GOODS_PRICE'] = app_train_domain['AMT_GOODS_PRICE'].isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})

app_train_domain['AMT_INCOME_TOTAL / DAYS_BIRTH']=app_train_domain['AMT_INCOME_TOTAL']/(app_train_domain['DAYS_BIRTH'])
app_train_domain['AMT_INCOME_TOTAL / DAYS_EMPLOYED']=app_train_domain['AMT_INCOME_TOTAL']/app_train_domain['DAYS_EMPLOYED']
app_train_domain['AMT_CREDIT / DAYS_BIRTH']=app_train_domain['AMT_CREDIT']/(app_train_domain['DAYS_BIRTH'])
app_train_domain['AMT_CREDIT / DAYS_EMPLOYED']=app_train_domain['AMT_CREDIT']/app_train_domain['DAYS_EMPLOYED']
app_train_domain['AMT_ANNUITY / DAYS_BIRTH']=app_train_domain['AMT_ANNUITY']/(app_train_domain['DAYS_BIRTH'])
app_train_domain['AMT_ANNUITY / DAYS_EMPLOYED']=app_train_domain['AMT_ANNUITY']/app_train_domain['DAYS_EMPLOYED']




app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / (1+app_test_domain['AMT_INCOME_TOTAL'])
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / (1+app_test_domain['AMT_INCOME_TOTAL'])
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['PRICE_INCOME_PERCENT'] = app_test_domain['AMT_GOODS_PRICE'] / (1+app_test_domain['AMT_INCOME_TOTAL'])
app_test_domain['CREDIT_PRICE_PERCENT'] = app_test_domain['AMT_CREDIT'] / (1+app_test_domain['AMT_GOODS_PRICE'])
app_test_domain['AMT_CREDIT - AMT_GOODS_PRICE'] = app_test_domain['AMT_CREDIT'] - app_test_domain['AMT_GOODS_PRICE']
app_test_domain['AMT_INCOME_TOTAL - AMT_ANNUITY'] = app_test_domain['AMT_INCOME_TOTAL'] - app_test_domain['AMT_ANNUITY']
app_test_domain['AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = app_test_domain['AMT_INCOME_TOTAL'] / 12. - app_test_domain['AMT_ANNUITY']
app_test_domain['AMT_INCOME_TOTAL - AMT_CREDIT'] = app_test_domain['AMT_INCOME_TOTAL'] - app_test_domain['AMT_CREDIT']
app_test_domain['AMT_INCOME_TOTAL / CREDIT_TERM - AMT_CREDIT'] = app_test_domain['AMT_INCOME_TOTAL']/app_test_domain['CREDIT_TERM'] - app_test_domain['AMT_CREDIT']
#app_test_domain['AMT_INCOME_TOTAL*5 - AMT_CREDIT'] = app_test_domain['AMT_INCOME_TOTAL']*5 - app_test_domain['AMT_CREDIT']
#app_test_domain['AMT_INCOME_TOTAL*10 - AMT_CREDIT'] = app_test_domain['AMT_INCOME_TOTAL']*10 - app_test_domain['AMT_CREDIT']
#app_test_domain['AMT_INCOME_TOTAL*20 - AMT_CREDIT'] = app_test_domain['AMT_INCOME_TOTAL']*20 - app_test_domain['AMT_CREDIT']
app_test_domain['AMT_GOODS_PRICE / AMT_ANNUITY'] = app_test_domain['AMT_GOODS_PRICE'] / app_test_domain['AMT_ANNUITY']
app_test_domain['AMT_GOODS_PRICE - AMT_ANNUITY'] = app_test_domain['AMT_GOODS_PRICE'] - app_test_domain['AMT_ANNUITY']
app_test_domain['AMT_GOODS_PRICE - AMT_INCOME_TOTAL'] = app_test_domain['AMT_GOODS_PRICE'] - app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['AMT_CREDIT - AMT_ANNUITY'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_ANNUITY']

app_test_domain['app DAYS_EMPLOYED - DAYS_BIRTH'] = app_test_domain['DAYS_EMPLOYED'] - app_test_domain['DAYS_BIRTH']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
app_test_domain['CAR_TO_BIRTH_RATIO_trick'] = (app_test_domain['OWN_CAR_AGE'] / app_test_domain['DAYS_BIRTH'])
app_test_domain['CAR_TO_BIRTH_RATIO'] = (app_test_domain['OWN_CAR_AGE'] / app_test_domain['DAYS_BIRTH'])*app_test_domain['FLAG_OWN_CAR_Y']
app_test_domain['PHONE_TO_BIRTH_RATIO'] = app_test_domain['DAYS_LAST_PHONE_CHANGE'] / app_test_domain['DAYS_BIRTH']
app_test_domain['CAR_TO_EMPLOY_RATIO_trick'] = (app_test_domain['OWN_CAR_AGE'] / app_test_domain['DAYS_EMPLOYED'])
app_test_domain['CAR_TO_EMPLOY_RATIO'] = (app_test_domain['OWN_CAR_AGE'] / app_test_domain['DAYS_EMPLOYED'])*app_test_domain['FLAG_OWN_CAR_Y']
app_test_domain['PHONE_TO_EMPLOY_RATIO'] = app_test_domain['DAYS_LAST_PHONE_CHANGE'] / app_test_domain['DAYS_EMPLOYED']

app_test_domain['THREE_SOURDES'] = app_test_domain['EXT_SOURCE_1'] * app_test_domain['EXT_SOURCE_2']* app_test_domain['EXT_SOURCE_3']
app_test_domain['app EXT_SOURCE std'] = app_test_domain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
app_test_domain['app EXT_SOURCE_1 * EXT_SOURCE_2'] = app_test_domain['EXT_SOURCE_1'] * app_test_domain['EXT_SOURCE_2']
app_test_domain['app EXT_SOURCE_1 * EXT_SOURCE_3'] = app_test_domain['EXT_SOURCE_1'] * app_test_domain['EXT_SOURCE_3']
app_test_domain['app EXT_SOURCE_2 * EXT_SOURCE_3'] = app_test_domain['EXT_SOURCE_2'] * app_test_domain['EXT_SOURCE_3']
app_test_domain['app EXT_SOURCE_1 * DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_1'] * app_test_domain['DAYS_BIRTH']
app_test_domain['app EXT_SOURCE_2 * DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_2'] * app_test_domain['DAYS_BIRTH']
app_test_domain['app EXT_SOURCE_3 * DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_3'] * app_test_domain['DAYS_BIRTH']
app_test_domain['app EXT_SOURCE_1 / DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_1'] / app_test_domain['DAYS_BIRTH']
app_test_domain['app EXT_SOURCE_2 / DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_2'] / app_test_domain['DAYS_BIRTH']
app_test_domain['app EXT_SOURCE_3 / DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_3'] / app_test_domain['DAYS_BIRTH']

app_test_domain['EXT_SOURCE_AVE / DAYS_BIRTH'] = app_test_domain['EXT_SOURCE_AVE'] / app_test_domain['DAYS_BIRTH']
app_test_domain['EXT_SOURCE_AVE / DAYS_REGISTRATION'] = app_test_domain['EXT_SOURCE_AVE'] / app_test_domain['DAYS_REGISTRATION']
app_test_domain['EXT_SOURCE_AVE / DAYS_ID_PUBLISH'] = app_test_domain['EXT_SOURCE_AVE'] / app_test_domain['DAYS_ID_PUBLISH']

app_test_domain['INCOME_PER_PERSON']=app_test_domain['AMT_INCOME_TOTAL']/(app_test_domain['CNT_FAM_MEMBERS'])
#app_test_domain['CHILDREN_TO_PARENTS_RATIO']=app_test_domain['CNT_CHILDREN']/(app_test_domain['CNT_PARENTS_MEMBERS'])
#app_test_domain['CHILDREN_TO_FAMILY_RATIO']=app_test_domain['CNT_CHILDREN']/(app_test_domain['CNT_FAM_MEMBERS'])
app_test_domain['INC_PER_CHLD']=app_test_domain['AMT_INCOME_TOTAL']/(1+app_test_domain['CNT_CHILDREN'])

app_test_domain['app most popular AMT_GOODS_PRICE'] = app_test_domain['AMT_GOODS_PRICE'].isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
app_test_domain['app popular AMT_GOODS_PRICE'] = app_test_domain['AMT_GOODS_PRICE'].isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})

app_test_domain['AMT_INCOME_TOTAL / DAYS_BIRTH']=app_test_domain['AMT_INCOME_TOTAL']/(app_test_domain['DAYS_BIRTH'])
app_test_domain['AMT_INCOME_TOTAL / DAYS_EMPLOYED']=app_test_domain['AMT_INCOME_TOTAL']/app_test_domain['DAYS_EMPLOYED']
app_test_domain['AMT_CREDIT / DAYS_BIRTH']=app_test_domain['AMT_CREDIT']/(app_test_domain['DAYS_BIRTH'])
app_test_domain['AMT_CREDIT / DAYS_EMPLOYED']=app_test_domain['AMT_CREDIT']/app_test_domain['DAYS_EMPLOYED']
app_test_domain['AMT_ANNUITY / DAYS_BIRTH']=app_test_domain['AMT_ANNUITY']/(app_test_domain['DAYS_BIRTH'])
app_test_domain['AMT_ANNUITY / DAYS_EMPLOYED']=app_test_domain['AMT_ANNUITY']/app_test_domain['DAYS_EMPLOYED']

'''
F_EXT_SOURCE_AVE = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==1,'EXT_SOURCE_AVE'].mean()
M_EXT_SOURCE_AVE = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==0,'EXT_SOURCE_AVE'].mean()
app_train_domain['EXT_SOURCE_AVE_GENDER']=app_train_domain['EXT_SOURCE_AVE']
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_AVE_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_AVE_GENDER']-F_EXT_SOURCE_AVE
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_AVE_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_AVE_GENDER']-M_EXT_SOURCE_AVE

F_EXT_SOURCE_AVE = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==1,'EXT_SOURCE_AVE'].mean()
M_EXT_SOURCE_AVE = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==0,'EXT_SOURCE_AVE'].mean()
app_test_domain['EXT_SOURCE_AVE_GENDER']=app_test_domain['EXT_SOURCE_AVE']
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_AVE_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_AVE_GENDER']-F_EXT_SOURCE_AVE
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_AVE_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_AVE_GENDER']-M_EXT_SOURCE_AVE

F_EXT_SOURCE_1 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==1,'EXT_SOURCE_1'].mean()
M_EXT_SOURCE_1 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==0,'EXT_SOURCE_1'].mean()
app_train_domain['EXT_SOURCE_1_GENDER']=app_train_domain['EXT_SOURCE_1']
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_1_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_1_GENDER']-F_EXT_SOURCE_1
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_1_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_1_GENDER']-M_EXT_SOURCE_1

F_EXT_SOURCE_1 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==1,'EXT_SOURCE_1'].mean()
M_EXT_SOURCE_1 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==0,'EXT_SOURCE_1'].mean()
app_test_domain['EXT_SOURCE_1_GENDER']=app_test_domain['EXT_SOURCE_1']
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_1_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_1_GENDER']-F_EXT_SOURCE_1
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_1_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_1_GENDER']-M_EXT_SOURCE_1

F_EXT_SOURCE_2 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==1,'EXT_SOURCE_2'].mean()
M_EXT_SOURCE_2 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==0,'EXT_SOURCE_2'].mean()
app_train_domain['EXT_SOURCE_2_GENDER']=app_train_domain['EXT_SOURCE_2']
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_2_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_2_GENDER']-F_EXT_SOURCE_2
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_2_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_2_GENDER']-M_EXT_SOURCE_2

F_EXT_SOURCE_2 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==1,'EXT_SOURCE_2'].mean()
M_EXT_SOURCE_2 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==0,'EXT_SOURCE_2'].mean()
app_test_domain['EXT_SOURCE_2_GENDER']=app_test_domain['EXT_SOURCE_2']
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_2_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_2_GENDER']-F_EXT_SOURCE_2
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_2_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_2_GENDER']-M_EXT_SOURCE_2

F_EXT_SOURCE_3 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==1,'EXT_SOURCE_3'].mean()
M_EXT_SOURCE_3 = app_train_domain.loc[app_train_domain['CODE_GENDER_F']==0,'EXT_SOURCE_3'].mean()
app_train_domain['EXT_SOURCE_3_GENDER']=app_train_domain['EXT_SOURCE_3']
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_3_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==1),'EXT_SOURCE_3_GENDER']-F_EXT_SOURCE_3
app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_3_GENDER']=app_train_domain.loc[(app_train_domain['CODE_GENDER_F']==0),'EXT_SOURCE_3_GENDER']-M_EXT_SOURCE_3

F_EXT_SOURCE_3 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==1,'EXT_SOURCE_3'].mean()
M_EXT_SOURCE_3 = app_test_domain.loc[app_test_domain['CODE_GENDER_F']==0,'EXT_SOURCE_3'].mean()
app_test_domain['EXT_SOURCE_3_GENDER']=app_test_domain['EXT_SOURCE_3']
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_3_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==1),'EXT_SOURCE_3_GENDER']-F_EXT_SOURCE_3
app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_3_GENDER']=app_test_domain.loc[(app_test_domain['CODE_GENDER_F']==0),'EXT_SOURCE_3_GENDER']-M_EXT_SOURCE_3
'''


print('Domain Training Features shape: ', app_train_domain.shape)
print('Domain Testing Features shape: ', app_test_domain.shape)
app_train_domain, app_test_domain = app_train_domain.align(app_test_domain, join = 'inner', axis = 1)
# Add the target back in
app_train_domain['TARGET'] = train_labels
print('Domain Training Features shape: ', app_train_domain.shape)
print('Domain Testing Features shape: ', app_test_domain.shape)

#app_train_domain['TARGET'] = train_labels
app_train_domain.to_csv('m_train_domain.csv', index = False)
app_test_domain.to_csv('m_test_domain.csv', index = False)

#lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    #k_fold = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 50)
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features, labels):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


#app_train_domain['TARGET'] = train_labels
# Test the domain knolwedge features
submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)

#fi_sorted = plot_feature_importances(fi_domain)

submission_domain.to_csv('baseline_lgb_domain_features.csv', index = False)
fi_domain.to_csv('baseline_lgb_domain_features_important.csv', index = False)
#添加 1，2，3，birth 相互乘除
#overall  0.823430  0.766733
#添加 cnt child / cnt parent 重要性低，删除
#overall  0.817453  0.766639 
#CHILDREN_TO_PARENTS_RATIO	2.6
#CHILDREN_TO_FAMILY_RATIO	0
#添加了 许多减法
#overall  0.825539  0.766637
#overall  0.821497  0.766651

# StratifiedKFold
# overall  0.815859  0.766456

# flag 
# overall  0.826247  0.767384

# 添加ave source 除以各种年龄相关
# overall  0.818789  0.767082

# income/annuity/credit age
# overall  0.823593  0.767362

# employed——day 男女不同 用clean v3 变好一点
#overall  0.817587  0.767408

#employed nan 没采用
#0.820 0.76700
#birth day -10不采用
#overall  0.822920  0.767235

#Mean encode, (gender) source 1,2,3,ave 用的减法，没有提升
#overall  0.817949  0.767335

# ext source  fill nan
# overall  0.824794  0.767554

#最后结果overall  0.817587  0.767408
