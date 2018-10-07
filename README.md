# 43th-solution
Kaggle: Home credit default risk

Goal: 
====================

predict whether a client will repay a loan or default 

Input:
====================

can be downloaded from [here](https://www.kaggle.com/c/home-credit-default-risk/data):

There are 7 different sources of data: 

application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits. 

bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length. 

previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV. 

POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows. 

credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows. 

installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

Baseline:
====================

My work starts from [Will Koehrsen’s kernel](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction): His code can get 0.782 ROC AUC 

My work gets 0.800 ROC AUC score. My improvement comes from: 

1. Add hundreds of new features during feature engineering 
2. Find two quite different LightGBM parameter sets can improve cv 
3. Add Xgboost and stack with those two LightGBM results

My code:
====================
#### 1_clean_test_train_v3

INPUT: **application_train.csv** and **application_test.csv**

Use plot/value_counts/describe/isnull to investigate features

Clean some outliers

Fill some nan by some meaningful values if i am sure about meaning of those features; 

OUTPUT: **train_cleaned.csv** and **test_cleaned.csv**

#### 2_feature_engineer_v8

INPUT: **train_cleaned.csv** and **test_cleaned.csv**

EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3 turn out to be 3 most important features. They are credit score from other bureau companies. There are some other important features and i use them + - * / each other to try to get some good features.

OUTPUT: **m_train_domain.csv** and **m_test_domain.csv**

