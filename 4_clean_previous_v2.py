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
# Memory management
import gc 


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

def agg_numeric_std(df, group_var, df_name):
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
    agg = numeric_df.groupby(group_var).agg([ 'mean', 'max', 'min', 'sum', 'std']).reset_index()
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
    categorical = categorical.groupby(group_var).agg(['sum', 'mean','count'])   
    column_names = []
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'mean','count']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    return categorical

#---------------------
cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')

cash_balance['MONTHS_END']=cash_balance['MONTHS_BALANCE']+cash_balance['CNT_INSTALMENT_FUTURE']
cash_balance_counts = count_categorical(cash_balance, group_var = 'SK_ID_PREV', df_name = 'cash')
cash_balance_counts.head()
cash_balance_agg = agg_numeric(cash_balance, group_var = 'SK_ID_PREV', df_name = 'cash')
cash_balance_agg.head()
# Dataframe grouped by the loan
cash_by_loan = cash_balance_agg.merge(cash_balance_counts,  right_index = True, left_on = 'SK_ID_PREV', how = 'outer')

#---------------------我加的
#我加的，删除没用的，增加最后一周的状态
col_to_drop = ['cash_MONTHS_BALANCE_mean','cash_MONTHS_BALANCE_sum','cash_CNT_INSTALMENT_sum',
               'cash_CNT_INSTALMENT_FUTURE_sum' , 'cash_MONTHS_END_sum','cash_SK_DPD_min',
               'cash_SK_DPD_DEF_min','cash_NAME_CONTRACT_STATUS_XNA_mean',
               'cash_NAME_CONTRACT_STATUS_XNA_sum','cash_NAME_CONTRACT_STATUS_Canceled_mean',
               'cash_NAME_CONTRACT_STATUS_Canceled_sum']
cash_by_loan=cash_by_loan[cash_by_loan.columns.drop(col_to_drop)]

cash_by_loan['cash_CNT_INSTALMENT_max - cash_CNT_INSTALMENT_min']=cash_by_loan['cash_CNT_INSTALMENT_max']-cash_by_loan['cash_CNT_INSTALMENT_min']
cash_by_loan['cash_CNT_INSTALMENT_var']=cash_by_loan['cash_CNT_INSTALMENT_max - cash_CNT_INSTALMENT_min']/cash_by_loan['cash_CNT_INSTALMENT_mean']

cash_by_loan['cash_CNT_INSTALMENT_FUTURE_max - cash_CNT_INSTALMENT_FUTURE_min']=cash_by_loan['cash_CNT_INSTALMENT_FUTURE_max']-cash_by_loan['cash_CNT_INSTALMENT_FUTURE_min']
cash_by_loan['cash_CNT_INSTALMENT_FUTURE_var']=cash_by_loan['cash_CNT_INSTALMENT_FUTURE_max - cash_CNT_INSTALMENT_FUTURE_min']/cash_by_loan['cash_CNT_INSTALMENT_FUTURE_mean']

cash_by_loan['cash_MONTHS_END_max - cash_MONTHS_END_min']=cash_by_loan['cash_MONTHS_END_max']-cash_by_loan['cash_MONTHS_END_min']
cash_by_loan['cash_MONTHS_END_var']=cash_by_loan['cash_MONTHS_END_max - cash_MONTHS_END_min']/cash_by_loan['cash_MONTHS_END_mean']

col_names=list(cash_by_loan.columns)
col_names[1]='MONTHS_BALANCE'
cash_by_loan.columns=col_names
a=cash_balance[['SK_ID_PREV','MONTHS_BALANCE','MONTHS_END','SK_DPD','SK_DPD_DEF']]
cash_by_loan=cash_by_loan.merge(a,on=['SK_ID_PREV','MONTHS_BALANCE'],how = 'left')
print(cash_by_loan.shape)
cash_by_loan.to_csv('cash_by_loan.csv', index = False)

#-----------------------
installments_payments = pd.read_csv('../input/installments_payments.csv')
installments_payments['DAYS_DELAY']=installments_payments['DAYS_INSTALMENT']-installments_payments['DAYS_ENTRY_PAYMENT']
installments_payments['AMOUNT_DIFF']=installments_payments['AMT_INSTALMENT']-installments_payments['AMT_PAYMENT']
installments_payments = agg_numeric_std(installments_payments, group_var = 'SK_ID_PREV', df_name = 'INSTALL')

col_to_drop = ['INSTALL_NUM_INSTALMENT_VERSION_sum','INSTALL_NUM_INSTALMENT_VERSION_std','INSTALL_NUM_INSTALMENT_NUMBER_sum',
               'INSTALL_NUM_INSTALMENT_NUMBER_std','INSTALL_DAYS_INSTALMENT_mean', 'INSTALL_DAYS_INSTALMENT_sum',
               'INSTALL_DAYS_INSTALMENT_std', 'INSTALL_DAYS_ENTRY_PAYMENT_sum' , 'INSTALL_DAYS_ENTRY_PAYMENT_std',
               'INSTALL_NUM_INSTALMENT_NUMBER_min'
               ]
installments_payments=installments_payments[installments_payments.columns.drop(col_to_drop)]
installments_payments['INSTALL_TERM']=installments_payments['INSTALL_DAYS_INSTALMENT_max']-installments_payments['INSTALL_DAYS_INSTALMENT_min']
installments_payments['INSTALL_ANNUITY']=installments_payments['INSTALL_AMT_INSTALMENT_sum']/installments_payments['INSTALL_TERM']*365
installments_payments['INSTALL_RATIO']=installments_payments['INSTALL_AMT_INSTALMENT_max']/installments_payments['INSTALL_AMT_INSTALMENT_min']

print(installments_payments.shape)
installments_payments.to_csv('installments_payments_by_loan.csv', index = False)
#------------------------


credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
credit_card_balance.loc[(credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']==0),'AMT_CREDIT_LIMIT_ACTUAL']=np.nan
credit_card_balance['AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL']=(10+credit_card_balance['AMT_BALANCE'])/(10+credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'])
credit_card_balance=credit_card_balance[(credit_card_balance['AMT_BALANCE']>-10000)]

credit_card_balance['Interest_rate']=(credit_card_balance['AMT_RECIVABLE']-credit_card_balance['AMT_RECEIVABLE_PRINCIPAL'])/credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']
credit_card_balance['Time_left']=credit_card_balance['AMT_TOTAL_RECEIVABLE']/credit_card_balance['AMT_PAYMENT_TOTAL_CURRENT']
credit_card_balance['Credit_end']=credit_card_balance['Time_left']+credit_card_balance['MONTHS_BALANCE']

credit_card_balance.loc[(credit_card_balance['NAME_CONTRACT_STATUS']!='Active'),'AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL']=np.nan
credit_card_balance.loc[(credit_card_balance['NAME_CONTRACT_STATUS']!='Active'),'Interest_rate']=np.nan
credit_card_balance.loc[(credit_card_balance['NAME_CONTRACT_STATUS']!='Active'),'Time_left']=np.nan
credit_card_balance.loc[(credit_card_balance['NAME_CONTRACT_STATUS']!='Active'),'Credit_end']=np.nan


'''
credit_card_balance['card missing'] = credit_card_balance.isnull().sum(axis = 1).values

credit_card_balance['card SK_DPD - MONTHS_BALANCE'] = credit_card_balance['SK_DPD'] - credit_card_balance['MONTHS_BALANCE']
credit_card_balance['card SK_DPD_DEF - MONTHS_BALANCE'] = credit_card_balance['SK_DPD_DEF'] - credit_card_balance['MONTHS_BALANCE']
credit_card_balance['card SK_DPD - SK_DPD_DEF'] = credit_card_balance['SK_DPD'] - credit_card_balance['SK_DPD_DEF']
    
credit_card_balance['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = credit_card_balance['AMT_TOTAL_RECEIVABLE'] - credit_card_balance['AMT_RECIVABLE']
credit_card_balance['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = credit_card_balance['AMT_TOTAL_RECEIVABLE'] - credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']
credit_card_balance['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = credit_card_balance['AMT_RECIVABLE'] - credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']

credit_card_balance['card AMT_BALANCE - AMT_RECIVABLE'] = credit_card_balance['AMT_BALANCE'] - credit_card_balance['AMT_RECIVABLE']
credit_card_balance['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = credit_card_balance['AMT_BALANCE'] - credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']
credit_card_balance['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = credit_card_balance['AMT_BALANCE'] - credit_card_balance['AMT_TOTAL_RECEIVABLE']

credit_card_balance['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] - credit_card_balance['AMT_DRAWINGS_ATM_CURRENT']
credit_card_balance['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] - credit_card_balance['AMT_DRAWINGS_OTHER_CURRENT']
credit_card_balance['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = credit_card_balance['AMT_DRAWINGS_CURRENT'] - credit_card_balance['AMT_DRAWINGS_POS_CURRENT']

#credit_card_balance=credit_card_balance[(credit_card_balance['NAME_CONTRACT_STATUS']!='Completed') ]
col_nams=['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT',
          'AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT','AMT_INST_MIN_REGULARITY',
          'AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_RECEIVABLE_PRINCIPAL',
          'AMT_RECIVABLE','AMT_TOTAL_RECEIVABLE','CNT_DRAWINGS_ATM_CURRENT',
          'CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT']
for col in col_nams:
    credit_card_balance.loc[(credit_card_balance['NAME_CONTRACT_STATUS']=='Completed') ,col]=np.nan
'''

#credit_card_balance_counts = count_categorical(credit_card_balance, group_var = 'SK_ID_PREV', df_name = 'credit')
#credit_card_balance_agg = agg_numeric(credit_card_balance, group_var = 'SK_ID_PREV', df_name = 'credit')
# Dataframe grouped by the loan
#creditcard_by_loan = credit_card_balance_agg.merge(credit_card_balance_counts,  right_index = True, left_on = 'SK_ID_PREV', how = 'outer')
creditcard_by_loan= agg_numeric(credit_card_balance, group_var = 'SK_ID_PREV', df_name = 'credit')
col_to_drop = ['credit_MONTHS_BALANCE_mean','credit_MONTHS_BALANCE_sum','credit_AMT_BALANCE_sum',
               'credit_SK_DPD_min','credit_SK_DPD_DEF_min','credit_Time_left_sum',
               'credit_Credit_end_sum','credit_Interest_rate_sum','credit_AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL_sum']
creditcard_by_loan=creditcard_by_loan[creditcard_by_loan.columns.drop(col_to_drop)]

'''
col_names=list(creditcard_by_loan.columns)
col_names[1]='MONTHS_BALANCE'
creditcard_by_loan.columns=col_names
a=credit_card_balance[['SK_ID_PREV','MONTHS_BALANCE','SK_DPD','SK_DPD_DEF']]
creditcard_by_loan=creditcard_by_loan.merge(a,on=['SK_ID_PREV','MONTHS_BALANCE'],how = 'left')

creditcard_by_loan.rename(columns={'SK_DPD': 'credit_SK_DPD'},inplace=True)
creditcard_by_loan.rename(columns={'SK_DPD_DEF': 'credit_SK_DPD_DEF'},inplace=True)
creditcard_by_loan.rename(columns={'MONTHS_BALANCE': 'credit_MONTHS_BALANCE'},inplace=True)
'''
print(creditcard_by_loan.shape)
creditcard_by_loan.to_csv('creditcard_by_loan.csv', index = False)


#----------------------------------------

previous = pd.read_csv('../input/previous_application.csv')
previous[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
             'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace = True)
previous['Previous_term']=previous['AMT_CREDIT']/previous['AMT_ANNUITY']
previous['Previous_term_v2']=previous['DAYS_TERMINATION']-previous['DAYS_FIRST_DUE']
previous['Days_ahead']=previous['DAYS_LAST_DUE_1ST_VERSION']-previous['DAYS_LAST_DUE']
previous['AMT_CREDIT - AMT_GOODS_PRICE']=previous['AMT_CREDIT']-previous['AMT_GOODS_PRICE']
previous['AMT_CREDIT / AMT_GOODS_PRICE']=previous['AMT_CREDIT']/previous['AMT_GOODS_PRICE']
previous['AMT_CREDIT - AMT_APPLICATION']=previous['AMT_CREDIT']-previous['AMT_APPLICATION']
previous['AMT_CREDIT / AMT_APPLICATION']=previous['AMT_CREDIT']/previous['AMT_APPLICATION']
previous['Previous_term_v2 / Previous_term']=previous['Previous_term_v2']/previous['Previous_term']


previous = previous.merge(cash_by_loan, on ='SK_ID_PREV', how = 'left')
previous = previous.merge(installments_payments, on ='SK_ID_PREV', how = 'left')
previous = previous.merge(creditcard_by_loan, on ='SK_ID_PREV', how = 'left')

# Calculate aggregate statistics for each numeric column
previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
print('Previous aggregation shape: ', previous_agg.shape)

# Calculate value counts for each categorical column
previous_counts = count_categorical(previous, 'SK_ID_CURR', 'previous')
print('Previous counts shape: ', previous_counts.shape)



#last 6 has more than 90% missing
col_to_drop = ['previous_Previous_term_sum','previous_HOUR_APPR_PROCESS_START_sum','previous_AMT_CREDIT / AMT_GOODS_PRICE_sum',
               'previous_AMT_CREDIT / AMT_APPLICATION_sum', 'previous_Previous_term_v2_sum', 'previous_DAYS_TERMINATION_sum',
               'previous_Previous_term_v2 / Previous_term_sum' , 'previous_RATE_INTEREST_PRIVILEGED_mean', 'previous_RATE_INTEREST_PRIVILEGED_max', 
               'previous_RATE_INTEREST_PRIMARY_mean', 'previous_RATE_INTEREST_PRIMARY_max', 'previous_RATE_INTEREST_PRIMARY_min',
               'previous_RATE_INTEREST_PRIVILEGED_min']

previous_agg=previous_agg[previous_agg.columns.drop(col_to_drop)]

train = pd.read_csv('./m_train_domain.csv')
test = pd.read_csv('./m_test_domain.csv')

# Merge in the previous information
train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')


#previous_AMT_ANNUITY_mean 79
train['previous_AMT_ANNUITY_mean / AMT_CREDIT']=train['previous_AMT_ANNUITY_mean']/train['AMT_CREDIT']
train['previous_AMT_ANNUITY_mean / AMT_ANNUITY']=train['previous_AMT_ANNUITY_mean']/train['AMT_ANNUITY']
train['previous_AMT_ANNUITY_mean / AMT_INCOME_TOTAL']=train['previous_AMT_ANNUITY_mean']/train['AMT_INCOME_TOTAL']

test['previous_AMT_ANNUITY_mean / AMT_CREDIT']=test['previous_AMT_ANNUITY_mean']/test['AMT_CREDIT']
test['previous_AMT_ANNUITY_mean / AMT_ANNUITY']=test['previous_AMT_ANNUITY_mean']/test['AMT_ANNUITY']
test['previous_AMT_ANNUITY_mean / AMT_INCOME_TOTAL']=test['previous_AMT_ANNUITY_mean']/test['AMT_INCOME_TOTAL']



# Function to calculate missing values by column# Funct 
def missing_values_table(df, print_info = False):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        if print_info:
            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
def remove_missing_columns(train, test, threshold = 90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)
    
    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)
    
    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    
    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    print(missing_columns)
    # Drop the missing columns and return
    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)
    
    return train, test

#train, test = remove_missing_columns(train, test)




gc.enable()
del creditcard_by_loan, installments_payments, cash_by_loan,previous,previous_agg, previous_counts
gc.collect()




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
        '''
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        '''
        ## Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.02, num_leaves=34,
                                   reg_alpha = 0.041545473, reg_lambda = .0735294, 
                                   subsample = .8715623, n_jobs = -1,nthread=4, colsample_bytree=.9497036,
                                   max_depth=8, min_split_gain=.0222415, min_child_weight=39.3259775, silent=-1,
                                   random_state = 50)

        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 200, verbose = 200)
        
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


submission, fi, metrics = model(train, test)
print(metrics)
submission.to_csv('submission_manualp2.csv', index = False)
fi.to_csv('submission_manualp2_importance.csv', index = False)
train.to_csv('train_previous_raw.csv', index = False)
test.to_csv('test_previous_raw.csv', index = False)

# train 0.847150 validation 0.779627 score 0.771

# use lastest domain v8 train/test
# overall  0.858728  0.782574

# install 和 cash加了些feature
# overall  0.855420  0.784043

#overall  0.852170  0.784568
#credit card limit 很多 0，替换为 nan
#overall  0.852092  0.784803

#不动credit card limit 很多 0，complete 行所有 0 替换为 nan 不行
#overall  0.849348  0.784295

#直接删除complete 不行
# overall  0.853603  0.784752

#complete 行所有 不管是不是0 替换为 nan 不行
#overall  0.849348  0.784295

#credit 加入最后月状态
#overall  0.853641  0.784326

#credit 加feature
#overall  0.854530  0.784666

#credit 加feature 并且非active改为nan
#overall  0.853129  0.784555

#previous 自己加feature
#overall  0.858241  0.784798

#previous annuity 比较 train, 换成lr 0.02的参数
#5  overall  0.883862  0.786898