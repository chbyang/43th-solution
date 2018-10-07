#https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection/notebook
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering

#import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

# modeling 
import lightgbm as lgb

# utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# memory management
import gc

# Read in data
train_small = pd.read_csv('./m_train_small_0.95.csv')
test_small = pd.read_csv('./m_test_small_0.95.csv')

train_labels = train_small['TARGET']


train_ids = train_small['SK_ID_CURR']
test_ids = test_small['SK_ID_CURR']

print('Training shape: ', train_small.shape)
print('Testing shape: ', test_small.shape)



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
    #k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    k_fold = StratifiedKFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    train_predictions = np.zeros(train_small.shape[0])
    train_predic = pd.DataFrame({'SK_ID_CURR': train_ids, 'TARGET': labels, 'PREDICT': train_predictions})
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features,labels):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        '''
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
        '''
        #first change parameters above to 
        #https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
        model = lgb.LGBMClassifier(
            nthread=12,
            n_estimators=50000,
            number_boosting_rounds=5000,
            learning_rate=0.001,
            max_bin=300,
            max_depth=-1,
            num_leaves=30,
            min_child_samples=70,
            subsample=1.0,
            subsample_freq=1,
            colsample_bytree=0.05,
            min_gain_to_split=0.5,
            reg_lambda=100,
            reg_alpha=0.0,
            scale_pos_weight=1,
            is_unbalance=False,
            min_child_weight=60,
            silent=-1,
            verbose=-1, )
        
        
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
        
        train_predic.ix[valid_indices,'PREDICT']=model.predict_proba(features[valid_indices])[:,1]
        print(train_predic['PREDICT'].isnull().sum())
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

    train_predic.to_csv('train_predic.csv', index = False)

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



submission_small, feature_importances_small, metrics_small = model(train_small, test_small)
print(metrics_small)
submission_small.to_csv('selected_features_small_submission.csv', index = False)
feature_importances_small.to_csv('feature_importances_small.csv', index = False)

#352 features
# train 0.865625 valid 0.784292 score 0.783

# stratified 
#726 features overall  0.903567  0.795382

#depth 6
#overall  0.883933  0.795426

'''
model = lgb.LGBMClassifier(
            nthread=12,
            n_estimators=10000,
            number_boosting_rounds=5000,
            learning_rate=0.02,
            max_bin=300,
            max_depth=-1,
            num_leaves=30,
            min_child_samples=70,
            subsample=1.0,
            subsample_freq=1,
            colsample_bytree=0.05,
            min_gain_to_split=0.5,
            reg_lambda=100,
            reg_alpha=0.0,
            scale_pos_weight=1,
            is_unbalance=False,
            min_child_weight=60,
            silent=-1,
            verbose=-1, )
      fold     train     valid
0        0  0.875208  0.795581
1        1  0.883140  0.799630
2        2  0.882991  0.795524
3        3  0.883948  0.796787
4        4  0.878076  0.800850
5  overall  0.880673  0.797651

0.005 lr
 fold     train     valid
0        0  0.869699  0.795957
1        1  0.887116  0.799938
2        2  0.879054  0.795381
3        3  0.884584  0.797212
4        4  0.882128  0.801232
5  overall  0.880516  0.797929

0.001 lr
valid's auc: 0.796128   train's auc: 0.873902
'''