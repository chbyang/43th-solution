# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:28:28 2018

@author: aa
"""
#https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg([ 'mean', 'max', 'min', 'sum']).reset_index()
    # Need to create new column names
    columns = [group_var]
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    return agg


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])   
    column_names = []  
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))    
    categorical.columns = column_names   
    return categorical


# Read in new copies of all the dataframes
train = pd.read_csv('./m_train_domain.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')

bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_counts.head()
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg.head()
# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')


#---------------------我加的
#我加的，删除没用的，增加最后一周的状态
bureau_by_loan.to_csv('bureau_by_loan.csv', index = False)
col_names=list(bureau_by_loan.columns)
col_names[3]='MONTHS_BALANCE'
bureau_by_loan.columns=col_names
bureau_by_loan=bureau_by_loan.merge(bureau_balance,on=['SK_ID_BUREAU','MONTHS_BALANCE'],how = 'left')
#col_to_drop = ['bureau_balance_MONTHS_BALANCE_count', 'bureau_balance_MONTHS_BALANCE_mean','bureau_balance_MONTHS_BALANCE_sum']
col_to_drop = ['bureau_balance_MONTHS_BALANCE_mean','bureau_balance_MONTHS_BALANCE_sum']

bureau_by_loan=bureau_by_loan[bureau_by_loan.columns.drop(col_to_drop)]
bureau_by_loan['bureau_balance_TARGET_count']=bureau_by_loan['bureau_balance_STATUS_2_count']+bureau_by_loan['bureau_balance_STATUS_3_count']+bureau_by_loan['bureau_balance_STATUS_4_count']+bureau_by_loan['bureau_balance_STATUS_5_count']
bureau_by_loan['bureau_balance_TARGET_norm']=bureau_by_loan['bureau_balance_STATUS_2_count_norm']+bureau_by_loan['bureau_balance_STATUS_3_count_norm']+bureau_by_loan['bureau_balance_STATUS_4_count_norm']+bureau_by_loan['bureau_balance_STATUS_5_count_norm']
col_to_drop=['bureau_balance_STATUS_2_count','bureau_balance_STATUS_3_count','bureau_balance_STATUS_4_count','bureau_balance_STATUS_5_count',
             'bureau_balance_STATUS_2_count_norm','bureau_balance_STATUS_3_count_norm','bureau_balance_STATUS_4_count_norm','bureau_balance_STATUS_5_count_norm',]
bureau_by_loan=bureau_by_loan[bureau_by_loan.columns.drop(col_to_drop)]
bureau_by_loan['bureau_balance_TARGET_norm']=bureau_by_loan['bureau_balance_TARGET_norm']/(1-bureau_by_loan['bureau_balance_STATUS_C_count_norm']-bureau_by_loan['bureau_balance_STATUS_X_count_norm'])
bureau_by_loan['bureau_balance_STATUS_0_count_norm']=bureau_by_loan['bureau_balance_STATUS_0_count_norm']/(1-bureau_by_loan['bureau_balance_STATUS_C_count_norm']-bureau_by_loan['bureau_balance_STATUS_X_count_norm'])
bureau_by_loan['bureau_balance_STATUS_1_count_norm']=bureau_by_loan['bureau_balance_STATUS_1_count_norm']/(1-bureau_by_loan['bureau_balance_STATUS_C_count_norm']-bureau_by_loan['bureau_balance_STATUS_X_count_norm'])
bureau_by_loan['bureau_balance_TARGET']=(bureau_by_loan['bureau_balance_TARGET_count']>0).astype('category')
bureau_by_loan['bureau_balance_TARGET_norm']=bureau_by_loan['bureau_balance_TARGET_norm'].fillna(value=0)
bureau_by_loan.to_csv('bureau_by_loan_recent.csv', index = False)
#-------------------
#bureau['CREDIT_ACTIVE'].value_counts()
#Closed      1079273
#Active       630607
#Sold           6527
#Bad debt         21
#bureau['CREDIT_CURRENCY'].value_counts()
#currency 1    1715020
#currency 2       1224
#currency 3        174
#currency 4         10
#很多标记active的已经关了，fix first
bureau.loc[ (bureau['DAYS_ENDDATE_FACT']<0) & (bureau['CREDIT_ACTIVE']=='Active'),'CREDIT_ACTIVE']='Closed'
bureau.loc[ (bureau['DAYS_CREDIT_ENDDATE']<0) & (bureau['CREDIT_ACTIVE']=='Active'),'CREDIT_ACTIVE']='Closed'
bureau.loc[bureau['DAYS_ENDDATE_FACT'].isnull(),'DAYS_ENDDATE_FACT']=bureau['DAYS_CREDIT_ENDDATE']
bureau.loc[bureau['DAYS_CREDIT_ENDDATE'].isnull(),'DAYS_CREDIT_ENDDATE']=bureau['DAYS_ENDDATE_FACT']

#how many day close before credit end, active account will be zero
bureau['DAY_ahead']=bureau['DAYS_ENDDATE_FACT']-bureau['DAYS_CREDIT_ENDDATE']
#how many day before close
bureau['DAY_left']=(bureau['DAYS_ENDDATE_FACT']>0)*bureau['DAYS_ENDDATE_FACT']

# amt_credit_sum 分为active 和不的
bureau['AMT_CREDIT_SUM_Active']=(bureau['CREDIT_ACTIVE']=='Active')*bureau['AMT_CREDIT_SUM']
bureau.loc[(bureau['CREDIT_ACTIVE']!='Active'),'AMT_CREDIT_SUM_Active']=np.nan
bureau.loc[(bureau['CREDIT_ACTIVE']=='Active'),'AMT_CREDIT_SUM']=np.nan


#
bureau['AMT_CREDIT_SUM_DEBT_Active']=(bureau['CREDIT_ACTIVE']=='Active')*bureau['AMT_CREDIT_SUM_DEBT']
bureau.loc[(bureau['CREDIT_ACTIVE']!='Active'),'AMT_CREDIT_SUM_DEBT_Active']=np.nan
bureau.loc[(bureau['CREDIT_ACTIVE']=='Active'),'AMT_CREDIT_SUM_DEBT']=np.nan
bureau.loc[(bureau['CREDIT_ACTIVE']=='Active')&(bureau[ 'AMT_CREDIT_SUM_DEBT_Active'].isnull()), 'AMT_CREDIT_SUM_DEBT_Active']=0.0

bureau.loc[(bureau['AMT_CREDIT_SUM_Active']==0)&(bureau['AMT_CREDIT_SUM_DEBT_Active']!=0),'AMT_CREDIT_SUM_Active']=bureau['AMT_CREDIT_SUM_DEBT_Active']

col_to_drop = ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT']
bureau=bureau[bureau.columns.drop(col_to_drop)]

bureau['Mortgage_SUM_CLOSED']=bureau['AMT_CREDIT_SUM']
bureau['Mortgage_SUM_ACTIVE']=bureau['AMT_CREDIT_SUM_Active']
bureau['Mortgage_DEBT_ACTIVE']=bureau['AMT_CREDIT_SUM_DEBT_Active']
bureau.loc[(bureau['CREDIT_TYPE']!='Mortgage'),'Mortgage_SUM_CLOSED']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Mortgage'),'Mortgage_SUM_ACTIVE']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Mortgage'),'Mortgage_DEBT_ACTIVE']=np.nan

#bureau['Microloan_SUM_CLOSED']=bureau['AMT_CREDIT_SUM']
#bureau['Microloan_SUM_ACTIVE']=bureau['AMT_CREDIT_SUM_Active']
#bureau['Microloan_DEBT_ACTIVE']=bureau['AMT_CREDIT_SUM_DEBT_Active']
#bureau.loc[(bureau['CREDIT_TYPE']!='Microloan'),'Microloan_SUM_CLOSED']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Microloan'),'Microloan_SUM_ACTIVE']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Microloan'),'Microloan_DEBT_ACTIVE']=np.nan

#bureau['Car loan_SUM_CLOSED']=bureau['AMT_CREDIT_SUM']
#bureau['Car loan_SUM_ACTIVE']=bureau['AMT_CREDIT_SUM_Active']
#bureau['Car loan_DEBT_ACTIVE']=bureau['AMT_CREDIT_SUM_DEBT_Active']
#bureau.loc[(bureau['CREDIT_TYPE']!='Car loan'),'Car loan_SUM_CLOSED']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Car loan'),'Car loan_SUM_ACTIVE']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Car loan'),'Car loan_DEBT_ACTIVE']=np.nan

bureau['Consumer credit_SUM_CLOSED']=bureau['AMT_CREDIT_SUM']
bureau['Consumer credit_SUM_ACTIVE']=bureau['AMT_CREDIT_SUM_Active']
bureau['Consumer credit_DEBT_ACTIVE']=bureau['AMT_CREDIT_SUM_DEBT_Active']
bureau.loc[(bureau['CREDIT_TYPE']!='Consumer credit'),'Consumer credit_SUM_CLOSED']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Consumer credit'),'Consumer credit_SUM_ACTIVE']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Consumer credit'),'Consumer credit_DEBT_ACTIVE']=np.nan

bureau['Credit card_SUM_CLOSED']=bureau['AMT_CREDIT_SUM']
bureau['Credit card_SUM_ACTIVE']=bureau['AMT_CREDIT_SUM_Active']
bureau['Credit card_DEBT_ACTIVE']=bureau['AMT_CREDIT_SUM_DEBT_Active']
bureau.loc[(bureau['CREDIT_TYPE']!='Credit card'),'Credit card_SUM_CLOSED']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Credit card'),'Credit card_SUM_ACTIVE']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Credit card'),'Credit card_DEBT_ACTIVE']=np.nan


#----------------------------

bureau['AMT_AVALABLE_Active']=bureau['AMT_CREDIT_SUM_Active']-bureau['AMT_CREDIT_SUM_DEBT_Active']
bureau['PER_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']/bureau['AMT_CREDIT_SUM_Active']


#bureau['Mortgage_AMT_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']
#bureau['Mortgage_PER_AVALABLE_Active']=bureau['PER_AVALABLE_Active']
#bureau.loc[(bureau['CREDIT_TYPE']!='Mortgage'),'Mortgage_AMT_AVALABLE_Active']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Mortgage'),'Mortgage_PER_AVALABLE_Active']=np.nan

#bureau['Microloan_AMT_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']
#bureau['Microloan_PER_AVALABLE_Active']=bureau['PER_AVALABLE_Active']
#bureau.loc[(bureau['CREDIT_TYPE']!='Microloan'),'Microloan_AMT_AVALABLE_Active']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Microloan'),'Microloan_PER_AVALABLE_Active']=np.nan

#bureau['Car loan_AMT_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']
#bureau['Car loan_PER_AVALABLE_Active']=bureau['PER_AVALABLE_Active']
#bureau.loc[(bureau['CREDIT_TYPE']!='Car loan'),'Car loan_AMT_AVALABLE_Active']=np.nan
#bureau.loc[(bureau['CREDIT_TYPE']!='Car loan'),'Car loan_PER_AVALABLE_Active']=np.nan

bureau['Consumer credit_AMT_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']
bureau['Consumer credit_PER_AVALABLE_Active']=bureau['PER_AVALABLE_Active']
bureau.loc[(bureau['CREDIT_TYPE']!='Consumer credit'),'Consumer credit_AMT_AVALABLE_Active']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Consumer credit'),'Consumer credit_PER_AVALABLE_Active']=np.nan

bureau['Credit card_AMT_AVALABLE_Active']=bureau['AMT_AVALABLE_Active']
bureau['Credit card_PER_AVALABLE_Active']=bureau['PER_AVALABLE_Active']
bureau.loc[(bureau['CREDIT_TYPE']!='Credit card'),'Credit card_AMT_AVALABLE_Active']=np.nan
bureau.loc[(bureau['CREDIT_TYPE']!='Credit card'),'Credit card_PER_AVALABLE_Active']=np.nan



bureau['MONTH_PAYMENT']=bureau['AMT_CREDIT_SUM_DEBT_Active']/bureau['DAY_left']*30
bureau['DAY_left_percent']=bureau['DAY_left']/(bureau['DAYS_CREDIT_ENDDATE']-bureau['DAYS_CREDIT'])

bureau['AMT_CREDIT_SUM_DEBT_Active_PERCENT']=bureau['AMT_CREDIT_SUM_DEBT_Active']*bureau['DAY_left_percent']
bureau['MONTH_PAYMENT_PERCENT']=bureau['MONTH_PAYMENT']*bureau['DAY_left_percent']


bureau = bureau.merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
bureau.to_csv('bureau_plus_bureau_balance.csv', index = False)

bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()




original_features = list(train.columns)
print('Original Number of Features: ', len(original_features))

# Merge with the value counts of bureau
train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
# Merge with the stats of bureau
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')




# Read in the test dataframe
test = pd.read_csv('./m_test_domain.csv')
# Merge with the value counts of bureau
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
# Merge with the stats of bureau
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
print('Shape of Testing Data: ', test.shape)


#======DAYS_EMPLOYED


#bureau_DAYS_CREDIT_max 44.6
train['bureau_DAYS_CREDIT_max / Day_birth']=train['bureau_DAYS_CREDIT_max']/train['DAYS_BIRTH']
train['bureau_DAYS_CREDIT_max / Day_ID']=train['bureau_DAYS_CREDIT_max']/train['DAYS_ID_PUBLISH']
train['bureau_DAYS_CREDIT_max / DAYS_EMPLOYED']=train['bureau_DAYS_CREDIT_max']/train['DAYS_EMPLOYED']
test['bureau_DAYS_CREDIT_max / Day_birth']=test['bureau_DAYS_CREDIT_max']/test['DAYS_BIRTH']
test['bureau_DAYS_CREDIT_max / Day_ID']=test['bureau_DAYS_CREDIT_max']/test['DAYS_ID_PUBLISH']
test['bureau_DAYS_CREDIT_max / DAYS_EMPLOYED']=test['bureau_DAYS_CREDIT_max']/test['DAYS_EMPLOYED']

train['bureau_DAYS_CREDIT_mean / Day_birth']=train['bureau_DAYS_CREDIT_mean']/train['DAYS_BIRTH']
train['bureau_DAYS_CREDIT_mean / Day_ID']=train['bureau_DAYS_CREDIT_mean']/train['DAYS_ID_PUBLISH']
train['bureau_DAYS_CREDIT_mean / DAYS_EMPLOYED']=train['bureau_DAYS_CREDIT_mean']/train['DAYS_EMPLOYED']
test['bureau_DAYS_CREDIT_mean / Day_birth']=test['bureau_DAYS_CREDIT_mean']/test['DAYS_BIRTH']
test['bureau_DAYS_CREDIT_mean / Day_ID']=test['bureau_DAYS_CREDIT_mean']/test['DAYS_ID_PUBLISH']
test['bureau_DAYS_CREDIT_mean / DAYS_EMPLOYED']=test['bureau_DAYS_CREDIT_mean']/test['DAYS_EMPLOYED']


#bureau_DAYS_CREDIT_ENDDATE_max 45.2
train['bureau_DAYS_CREDIT_ENDDATE_max / Day_birth']=train['bureau_DAYS_CREDIT_ENDDATE_max']/train['DAYS_BIRTH']
train['bureau_DAYS_CREDIT_ENDDATE_max / Day_ID']=train['bureau_DAYS_CREDIT_ENDDATE_max']/train['DAYS_ID_PUBLISH']
train['bureau_DAYS_CREDIT_ENDDATE_max / DAYS_EMPLOYED']=train['bureau_DAYS_CREDIT_ENDDATE_max']/train['DAYS_EMPLOYED']
test['bureau_DAYS_CREDIT_ENDDATE_max / Day_birth']=test['bureau_DAYS_CREDIT_ENDDATE_max']/test['DAYS_BIRTH']
test['bureau_DAYS_CREDIT_ENDDATE_max / Day_ID']=test['bureau_DAYS_CREDIT_ENDDATE_max']/test['DAYS_ID_PUBLISH']
test['bureau_DAYS_CREDIT_ENDDATE_max / DAYS_EMPLOYED']=test['bureau_DAYS_CREDIT_ENDDATE_max']/test['DAYS_EMPLOYED']

#bureau_AMT_CREDIT_MAX_OVERDUE_mean 49
train['bureau_AMT_CREDIT_MAX_OVERDUE_mean / AMT_CREDIT']=train['bureau_AMT_CREDIT_MAX_OVERDUE_mean']/train['AMT_CREDIT']
train['bureau_AMT_CREDIT_MAX_OVERDUE_mean / AMT_ANNUITY']=train['bureau_AMT_CREDIT_MAX_OVERDUE_mean']/train['AMT_ANNUITY']
train['bureau_AMT_CREDIT_MAX_OVERDUE_mean - AMT_CREDIT']=train['bureau_AMT_CREDIT_MAX_OVERDUE_mean']-train['AMT_CREDIT']
train['bureau_AMT_CREDIT_MAX_OVERDUE_mean - AMT_ANNUITY']=train['bureau_AMT_CREDIT_MAX_OVERDUE_mean']-train['AMT_ANNUITY']

test['bureau_AMT_CREDIT_MAX_OVERDUE_mean / AMT_CREDIT']=test['bureau_AMT_CREDIT_MAX_OVERDUE_mean']/test['AMT_CREDIT']
test['bureau_AMT_CREDIT_MAX_OVERDUE_mean / AMT_ANNUITY']=test['bureau_AMT_CREDIT_MAX_OVERDUE_mean']/test['AMT_ANNUITY']
test['bureau_AMT_CREDIT_MAX_OVERDUE_mean - AMT_CREDIT']=test['bureau_AMT_CREDIT_MAX_OVERDUE_mean']-test['AMT_CREDIT']
test['bureau_AMT_CREDIT_MAX_OVERDUE_mean - AMT_ANNUITY']=test['bureau_AMT_CREDIT_MAX_OVERDUE_mean']-test['AMT_ANNUITY']

#bureau_AMT_CREDIT_SUM_mean 48.8
train['bureau_AMT_CREDIT_SUM_mean / AMT_CREDIT']=train['bureau_AMT_CREDIT_SUM_mean']/train['AMT_CREDIT']
train['bureau_AMT_CREDIT_SUM_mean / AMT_ANNUITY']=train['bureau_AMT_CREDIT_SUM_mean']/train['AMT_ANNUITY']
train['bureau_AMT_CREDIT_SUM_mean - AMT_CREDIT']=train['bureau_AMT_CREDIT_SUM_mean']-train['AMT_CREDIT']
train['bureau_AMT_CREDIT_SUM_mean - AMT_ANNUITY']=train['bureau_AMT_CREDIT_SUM_mean']-train['AMT_ANNUITY']
test['bureau_AMT_CREDIT_SUM_mean / AMT_CREDIT']=test['bureau_AMT_CREDIT_SUM_mean']/test['AMT_CREDIT']
test['bureau_AMT_CREDIT_SUM_mean / AMT_ANNUITY']=test['bureau_AMT_CREDIT_SUM_mean']/test['AMT_ANNUITY']
test['bureau_AMT_CREDIT_SUM_mean - AMT_CREDIT']=test['bureau_AMT_CREDIT_SUM_mean']-test['AMT_CREDIT']
test['bureau_AMT_CREDIT_SUM_mean - AMT_ANNUITY']=test['bureau_AMT_CREDIT_SUM_mean']-test['AMT_ANNUITY']


#bureau_DAY_ahead_mean 43.6
train['bureau_DAY_ahead_mean / Day_birth']=train['bureau_DAY_ahead_mean']/train['DAYS_BIRTH']
train['bureau_DAY_ahead_mean / Day_ID']=train['bureau_DAY_ahead_mean']/train['DAYS_ID_PUBLISH']
train['bureau_DAY_ahead_mean / DAYS_EMPLOYED']=train['bureau_DAY_ahead_mean']/train['DAYS_EMPLOYED']
test['bureau_DAY_ahead_mean / Day_birth']=test['bureau_DAY_ahead_mean']/test['DAYS_BIRTH']
test['bureau_DAY_ahead_mean / Day_ID']=test['bureau_DAY_ahead_mean']/test['DAYS_ID_PUBLISH']
test['bureau_DAY_ahead_mean / DAYS_EMPLOYED']=test['bureau_DAY_ahead_mean']/test['DAYS_EMPLOYED']

#bureau_Mortgage_SUM_ACTIVE_sum 16.6
train['bureau_Mortgage_SUM_ACTIVE_sum / AMT_CREDIT']=train['bureau_Mortgage_SUM_ACTIVE_sum']/train['AMT_CREDIT']
train['bureau_Mortgage_SUM_ACTIVE_sum / AMT_ANNUITY']=train['bureau_Mortgage_SUM_ACTIVE_sum']/train['AMT_ANNUITY']
train['bureau_Mortgage_SUM_ACTIVE_sum / DAYS_BIRTH']=train['bureau_Mortgage_SUM_ACTIVE_sum']/train['DAYS_BIRTH']
train['bureau_Mortgage_SUM_ACTIVE_sum / DAYS_ID_PUBLISH']=train['bureau_Mortgage_SUM_ACTIVE_sum']/train['DAYS_ID_PUBLISH']
train['bureau_Mortgage_SUM_ACTIVE_sum - AMT_CREDIT']=train['bureau_Mortgage_SUM_ACTIVE_sum']-train['AMT_CREDIT']
train['bureau_Mortgage_SUM_ACTIVE_sum - AMT_ANNUITY']=train['bureau_Mortgage_SUM_ACTIVE_sum']-train['AMT_ANNUITY']

test['bureau_Mortgage_SUM_ACTIVE_sum / AMT_CREDIT']=test['bureau_Mortgage_SUM_ACTIVE_sum']/test['AMT_CREDIT']
test['bureau_Mortgage_SUM_ACTIVE_sum / AMT_ANNUITY']=test['bureau_Mortgage_SUM_ACTIVE_sum']/test['AMT_ANNUITY']
test['bureau_Mortgage_SUM_ACTIVE_sum / DAYS_BIRTH']=test['bureau_Mortgage_SUM_ACTIVE_sum']/test['DAYS_BIRTH']
test['bureau_Mortgage_SUM_ACTIVE_sum / DAYS_ID_PUBLISH']=test['bureau_Mortgage_SUM_ACTIVE_sum']/test['DAYS_ID_PUBLISH']
test['bureau_Mortgage_SUM_ACTIVE_sum - AMT_CREDIT']=test['bureau_Mortgage_SUM_ACTIVE_sum']-test['AMT_CREDIT']
test['bureau_Mortgage_SUM_ACTIVE_sum - AMT_ANNUITY']=test['bureau_Mortgage_SUM_ACTIVE_sum']-test['AMT_ANNUITY']


#48.8 bureau_Consumer credit_SUM_ACTIVE_sum
train['bureau_Consumer credit_SUM_ACTIVE_sum / AMT_CREDIT']=train['bureau_Consumer credit_SUM_ACTIVE_sum']/train['AMT_CREDIT']
train['bureau_Consumer credit_SUM_ACTIVE_sum / AMT_ANNUITY']=train['bureau_Consumer credit_SUM_ACTIVE_sum']/train['AMT_ANNUITY']
test['bureau_Consumer credit_SUM_ACTIVE_sum / AMT_CREDIT']=test['bureau_Consumer credit_SUM_ACTIVE_sum']/test['AMT_CREDIT']
test['bureau_Consumer credit_SUM_ACTIVE_sum / AMT_ANNUITY']=test['bureau_Consumer credit_SUM_ACTIVE_sum']/test['AMT_ANNUITY']

#bureau_AMT_AVALABLE_Active_sum 11.6
train['bureau_AMT_AVALABLE_Active_sum / AMT_CREDIT']=train['bureau_AMT_AVALABLE_Active_sum']/train['AMT_CREDIT']
train['bureau_AMT_AVALABLE_Active_sum - AMT_CREDIT']=train['bureau_AMT_AVALABLE_Active_sum']-train['AMT_CREDIT']
train['bureau_AMT_AVALABLE_Active_sum / AMT_ANNUITY']=train['bureau_AMT_AVALABLE_Active_sum']/train['AMT_ANNUITY']
train['bureau_AMT_AVALABLE_Active_sum - AMT_ANNUITY']=train['bureau_AMT_AVALABLE_Active_sum']-train['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_sum / AMT_CREDIT']=test['bureau_AMT_AVALABLE_Active_sum']/test['AMT_CREDIT']
test['bureau_AMT_AVALABLE_Active_sum / AMT_ANNUITY']=test['bureau_AMT_AVALABLE_Active_sum']/test['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_sum - AMT_ANNUITY']=test['bureau_AMT_AVALABLE_Active_sum']-test['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_sum - AMT_CREDIT']=test['bureau_AMT_AVALABLE_Active_sum']-test['AMT_CREDIT']


#bureau_AMT_AVALABLE_Active_min 34.8
train['bureau_AMT_AVALABLE_Active_min / AMT_CREDIT']=train['bureau_AMT_AVALABLE_Active_min']/train['AMT_CREDIT']
train['bureau_AMT_AVALABLE_Active_min / AMT_ANNUITY']=train['bureau_AMT_AVALABLE_Active_min']/train['AMT_ANNUITY']
train['bureau_AMT_AVALABLE_Active_min - AMT_ANNUITY']=train['bureau_AMT_AVALABLE_Active_min']-train['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_min / AMT_CREDIT']=test['bureau_AMT_AVALABLE_Active_min']/test['AMT_CREDIT']
test['bureau_AMT_AVALABLE_Active_min / AMT_ANNUITY']=test['bureau_AMT_AVALABLE_Active_min']/test['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_min - AMT_ANNUITY']=test['bureau_AMT_AVALABLE_Active_min']-test['AMT_ANNUITY']

#bureau_PER_AVALABLE_Active_min	119.6
train['bureau_PER_AVALABLE_Active_min / Day_birth']=train['bureau_PER_AVALABLE_Active_min']/train['DAYS_BIRTH']
train['bureau_PER_AVALABLE_Active_min / Day_ID']=train['bureau_PER_AVALABLE_Active_min']/train['DAYS_ID_PUBLISH']
test['bureau_PER_AVALABLE_Active_min / Day_birth']=test['bureau_PER_AVALABLE_Active_min']/test['DAYS_BIRTH']
test['bureau_PER_AVALABLE_Active_min / Day_ID']=test['bureau_PER_AVALABLE_Active_min']/test['DAYS_ID_PUBLISH']

#bureau_AMT_AVALABLE_Active_mean	25
train['bureau_AMT_AVALABLE_Active_mean / AMT_CREDIT']=train['bureau_AMT_AVALABLE_Active_mean']/train['AMT_CREDIT']
train['bureau_AMT_AVALABLE_Active_mean / AMT_ANNUITY']=train['bureau_AMT_AVALABLE_Active_mean']/train['AMT_ANNUITY']
test['bureau_AMT_AVALABLE_Active_mean / AMT_CREDIT']=test['bureau_AMT_AVALABLE_Active_mean']/test['AMT_CREDIT']
test['bureau_AMT_AVALABLE_Active_mean / AMT_ANNUITY']=test['bureau_AMT_AVALABLE_Active_mean']/test['AMT_ANNUITY']

#bureau_MONTH_PAYMENT_sum	20.8
train['bureau_MONTH_PAYMENT_sum / AMT_CREDIT']=train['bureau_MONTH_PAYMENT_sum']/train['AMT_CREDIT']
train['bureau_MONTH_PAYMENT_sum / AMT_ANNUITY']=train['bureau_MONTH_PAYMENT_sum']/train['AMT_ANNUITY']
test['bureau_MONTH_PAYMENT_sum / AMT_CREDIT']=test['bureau_MONTH_PAYMENT_sum']/test['AMT_CREDIT']
test['bureau_MONTH_PAYMENT_sum / AMT_ANNUITY']=test['bureau_MONTH_PAYMENT_sum']/test['AMT_ANNUITY']

train['bureau plus appl annuity / AMT_INCOME_TOTAL']=(train['AMT_ANNUITY']+12*train['bureau_MONTH_PAYMENT_sum'])/train['AMT_INCOME_TOTAL']
test['bureau plus appl annuity / AMT_INCOME_TOTAL']=(test['AMT_ANNUITY']+12*test['bureau_MONTH_PAYMENT_sum'])/test['AMT_INCOME_TOTAL']



train_labels = train['TARGET']
# Align the dataframes, this will remove the 'TARGET' column
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

print('Training Data Shape: ', train.shape)
print('Testing Data Shape: ', test.shape)

train.to_csv('train_bureau_raw.csv', index = False)
test.to_csv('test_bureau_raw.csv', index = False)
# drop 0 columns



import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc

import matplotlib.pyplot as plt

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
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
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
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        '''
        model = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
        '''
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

def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df



submission_raw, fi_raw, metrics_raw = model(train, test)
print(metrics_raw)    
fi_raw_sorted = plot_feature_importances(fi_raw)
submission_raw.to_csv('test_one.csv', index = False)
fi_raw_sorted.to_csv('fi_raw_sorted.csv', index = False)
fi_raw.to_csv('fi_raw.csv', index = False)

# bureau balance 提取最近一个月的状态
# overall  0.844835  0.772264

# bureau balance 增加 2，3，4，5违约的和
# overall  0.847031  0.772414

# bureau balance 删除 2，3，4，5 只保留他们的和
#  overall  0.844771  0.772457

#添加bureau['DAY_left']  bureau['MONTH_PAYMENT']
# bureau['DAY_left_percent'] bureau['AVALABLE'] 
# bureau['AVALABLE_percent']
#overall  0.846521  0.775166

#分离close和 active
#overall  0.841271  0.775144

#bureau delete two columns
#overall  0.848676  0.775548

# active debt fill 0
#overall  0.851361  0.775667
#learning low:overall  0.853840  0.778342

#加入var
#overall  0.839511  0.775598

#增加，morgate，car，microloan分类
# overall  0.847136  0.775659

#增加，morgate，car，microloan分类的 available available_percent
#  overall  0.837639  0.775825

#和train/test结合，没有特牛的features
# overall  0.842579  0.776026

#加了几个和annuity credit 相减
# overall  0.842085  0.776157

#删除了count var features size 572 反倒提高了
# overall  0.838009  0.776162
#比v2好，v2又尝试了些女的的kernal