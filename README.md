# 43th-solution
Kaggle: Home credit default risk

# Goal: 

predict whether a client will repay a loan or default 

# Input:

can be downloaded from [here](https://www.kaggle.com/c/home-credit-default-risk/data):

There are 7 different sources of data: 

application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits. 

bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length. 

previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV. 

POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows. 

credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows. 

installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

# Baseline:

My work starts from [Will Koehrsen’s kernel](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction): His code can get 0.782 ROC AUC 

My work gets 0.800 ROC AUC score. My improvement comes from: 

1. Add hundreds of new features during feature engineering 
2. Find two quite different LightGBM parameter sets can improve cv 
3. Add Xgboost and stack with those two LightGBM results

# My code:

## 1_clean_test_train_v3

INPUT: **application_train.csv** and **application_test.csv**

Use plot/value_counts/describe/isnull to investigate features

Clean some outliers

Fill some nan by some meaningful values if i am sure about meaning of those features; 

OUTPUT: **train_cleaned.csv** and **test_cleaned.csv**

## 2_feature_engineer_v8

INPUT: **train_cleaned.csv** and **test_cleaned.csv**

EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3 turn out to be 3 most important features. They are credit score from other bureau companies. There are some other important features and i use them + - * / each other to try to get some good features.

OUTPUT: **m_train_domain.csv** and **m_test_domain.csv**

## 3_clean_bureau_v2

INPUT: **m_train_domain.csv** and **m_test_domain.csv** and **bureau.csv** and **bureau_balance.csv**

m_train/test_domain have information of current application. bureau and bureau_balance are past applications. group them together and did more feature engineering.

OUTPUT: **train_bureau_raw.csv** and **test_bureau_raw.csv**

## 4_clean_previous_v2

INPUT: **m_train_domain.csv** and **m_test_domain.csv** and **previous_application.csv** and **POS_CASH_balance.csv** and **installments_payments.csv**

m_train/test_domain have information of current application. previous_application, POS_CASH_balance and installments_payments are past applications to HOME CREDIT. group them together and did more feature engineering.

OUTPUT: **train_previous_raw.csv** and **test_previous_raw.csv**

## 5_merge_test_train_bureau_previous

INPUT: **train_previous_raw.csv** and **test_previous_raw.csv** and **train_bureau_raw.csv** and **test_bureau_raw.csv**

Merge file got from 3 and 4

OUTPUT: **m_train_selected.csv** and **m_test_selected.csv**

## 6_feature_selection

INPUT: **m_train_selected.csv** and **m_test_selected.csv**

Run lgb model and output importance. onnly keep features that sum of whose normalized importance is 0.95

OUTPUT: **m_train_small.csv** and **m_test_small.csv**

## 7_lgb

INPUT: **m_train_small.csv** and **m_test_small.csv**

parameter_1
lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
                                   
cv=0.7956

parameter_2
lgb.LGBMClassifier(nthread=12,n_estimators=50000,number_boosting_rounds=5000,
            learning_rate=0.001,max_bin=300,max_depth=-1,num_leaves=30,min_child_samples=70,
            subsample=1.0,subsample_freq=1,colsample_bytree=0.05,min_gain_to_split=0.5,
            reg_lambda=100,reg_alpha=0.0,scale_pos_weight=1,is_unbalance=False,min_child_weight=60,
            silent=-1,verbose=-1, )
            
cv=0.7979

## 8_xgb

INPUT: **m_train_small.csv** and **m_test_small.csv**

cv=0.7968

## 9_stacking

suggest 0.4 xgb+0.3 lgb_1+0.3 lgb_2
or neural network stacking
