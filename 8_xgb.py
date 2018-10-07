import pandas as pd
import numpy as np
import xgboost as xgb
import gc
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

X=pd.read_csv('m_train_small_0.95.csv')
Y=X['TARGET'].values
#X_predict=X['TARGET']
#X_predict=pd.DataFrame(X_predict)
#X_predict['PREDICT']=np.nan
X=X.drop('TARGET',axis=1)
features=X.columns
X=X.fillna(-99999.0).values

X_test=pd.read_csv('m_test_small_0.95.csv').fillna(-99999.0).values
sub=pd.read_csv('../input/sample_submission.csv')

Y_test=np.zeros(X_test.shape[0])
feature_importance=np.zeros((features.shape))
ave_score=0.0
best_n_estimators=[]
'''
model=xgb.XGBClassifier(tree_method='gpu_hist',gpu_id = 0,booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0, 
                        learning_rate=0.02,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=None,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=False)#, eval_metric='auc', 
                        #early_stopping_rounds=300)#, nrounds=20000)


model=xgb.XGBClassifier(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.022,  max_depth=5,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
'''

model=xgb.XGBClassifier(booster='gbtree',
                        tree_method='gpu_hist',
                        gpu_id = 0, 
                        objective='binary:logistic', 
                        learning_rate=0.012,
                        max_leaves=40,
                        max_depth=16, 
                        max_bin=255,
                        subsample=0.5,
                        colsample_bytree=0.5, 
                        colsample_bylevel=1, 
                        min_child_weight=4, 
                        reg_alpha=0.001,
                        reg_lambda=0.001,
                        scale_pos_weight=1,
                        seed=1,
                        verbose_eval=100, 
                        n_estimators=10000,
                        silent=1 )

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

i=0
for train_index, val_index in kf.split( X, Y):
    i+=1
    if i!=4:
        continue
    #xgb_model = model.fit(X[train_index], Y[train_index])
    X_train=X[train_index]
    Y_train=Y[train_index]
    X_val=X[val_index]
    Y_val=Y[val_index]
    
    xgb_model = model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          eval_metric='auc', early_stopping_rounds = 100)
    print('Best n_estimators',xgb_model.best_ntree_limit)
    best_n_estimators.append(xgb_model.best_ntree_limit)
    
    ave_score+=xgb_model.best_score/5
    Y_test+=xgb_model.predict_proba(X_test)[:,1]/5
    print(len(Y_test))
    feature_importance+=xgb_model.feature_importances_ /5
    pd.DataFrame(Y_test).to_csv('Y_test'+str(i)+'_'+str(xgb_model.best_score)+'.csv',index=False)
    pd.DataFrame(feature_importance).to_csv('feature_importance'+str(i)+'.csv',index=False)
    del X_train,Y_train,Y_val
    gc.collect()
    #X_predict.ix[val_index,'PREDICT']=xgb_model.predict_proba(X_val)[:,1]
    #print(X_predict['PREDICT'].isnull().sum())
    #X_predict.to_csv('X_predict'+str(i)+'.csv',index=False)
    del X_val, xgb_model
    gc.collect()
        
fea_imp=pd.DataFrame({'features': features, 'importance': feature_importance})
#fea_imp=fea_imp.sort_values(by=['importance'])
print('cv= ',ave_score)
sub['TARGET']=Y_test
print(Y_test.sum())
print(best_n_estimators)
sub.to_csv('xgb_prediction.csv',index=False)
pd.DataFrame(fea_imp).to_csv('xgb_importance.csv',index=False)

#train_predic=pd.DataFrame({'TARGET': Y, 'PREDICT': X_predict})
#train_predic.to_csv('train_predic.csv',index=False)
'''
(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
#cv= lr 0.05  0.7942924

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0, 
                        learning_rate=0.02,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
#cv= lr 0.02  0.7954152

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=20, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=111,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
#cv=  0.7941047

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=40, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
#cv=0.7941204

model=xgb.XGBClassifier(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
cv0.7943754

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.05, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
0.7943754

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.03, 
                        learning_rate=0.05,  max_depth=6,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
0.7943754
(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.05,  max_depth=9,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
cv=  0.79199

(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.05,  max_depth=7,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
 0.7934506
(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.05,  max_depth=5,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
 0.7948342
(tree_method='gpu_hist',gpu_id = 0, booster='gbtree', 
                        colsample_bylevel=0.632, colsample_bytree=0.7, gamma=0.01, 
                        learning_rate=0.05,  max_depth=4,
                        min_child_weight=30, objective='binary:logistic', 
                        reg_alpha=0, reg_lambda=0, seed=1,
                        subsample=0.85, verbose_eval=100, n_estimators=4000,
                        silent=1 )
early 200
 cv=  0.794676
 
 model=xgb.XGBClassifier(booster='gbtree',
                        tree_method='gpu_hist',
                        gpu_id = 0, 
                        objective='binary:logistic', 
                        learning_rate=0.001,
                        max_leaves=40,
                        max_depth=16, 
                        max_bin=255,
                        subsample=0.5,
                        colsample_bytree=0.5, 
                        colsample_bylevel=1, 
                        min_child_weight=4, 
                        reg_alpha=0.001,
                        reg_lambda=0.001,
                        scale_pos_weight=1,
                        seed=1,
                        verbose_eval=100, 
                        n_estimators=10000,
                        silent=1 )
 
 
'''