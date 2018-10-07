import numpy as np
import pandas as pd 

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt

app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')

#print(app_train['NAME_CONTRACT_TYPE'].value_counts())
#print(app_test['NAME_CONTRACT_TYPE'].value_counts())



print(len(app_train))
app_train=app_train[app_train['AMT_INCOME_TOTAL']<=18000090]
print(len(app_train))
#overall  0.803375  0.758537  score 0.735

#use train cut income below 4,410,000.  9 lines in train
#overall  0.806988  0.758736

#use train cut income below 18000090.  1 line in train
#overall  0.801842  0.758931
#最终决定。只删除那行income大于18000090的


app_train['AMT_ANNUITY'].plot.hist();
plt.show()
app_test['AMT_ANNUITY'].plot.hist();
plt.show()
'''
#print(app_test['AMT_ANNUITY'].value_counts())
print(app_train['AMT_ANNUITY'].describe())
print(app_train['AMT_CREDIT'].describe())
print(app_train['AMT_INCOME_TOTAL'].describe())
print(app_train['AMT_GOODS_PRICE'].describe())
'''
chosen_train=app_train['AMT_ANNUITY'].isnull()
no_annuity=app_train[chosen_train]
no_annuity.to_csv('train_No annuity.csv',index=False)

'''
print(app_test['AMT_ANNUITY'].describe())
print(app_test['AMT_CREDIT'].describe())
print(app_test['AMT_INCOME_TOTAL'].describe())
print(app_test['AMT_GOODS_PRICE'].describe())
'''
chosen_test=app_test['AMT_ANNUITY'].isnull()
no_annuity=app_test[chosen_test]
no_annuity.to_csv('test_No annuity.csv',index=False)

credit_annuity_ratio_train=app_train['AMT_CREDIT'].mean()/app_train['AMT_ANNUITY'].mean()
credit_annuity_ratio_test=app_test['AMT_CREDIT'].mean()/app_test['AMT_ANNUITY'].mean()

print(credit_annuity_ratio_train)
print(credit_annuity_ratio_test)

app_train.loc[chosen_train,'AMT_ANNUITY']=app_train.loc[chosen_train,'AMT_CREDIT']/credit_annuity_ratio_train
no_annuity=app_train[chosen_train]
#no_annuity.to_csv('train_No annuity_fixed.csv',index=False)

app_test.loc[chosen_test,'AMT_ANNUITY']=app_test.loc[chosen_test,'AMT_CREDIT']/credit_annuity_ratio_test
no_annuity=app_test[chosen_test]
#no_annuity.to_csv('test_No annuity_fixed.csv',index=False)


#有一些提升。加domain的提升较少
#v4 change from v3 fix annuity nan
#5  overall  0.806538  0.759119 


print(app_train['CODE_GENDER'].value_counts())
#train 有性别XNA 但是test没有。 删掉
app_train=app_train[app_train['CODE_GENDER']!='XNA']
print(app_train['CODE_GENDER'].value_counts())

'''
large_CHILD_train=app_train[app_train['CNT_CHILDREN']>=10]
large_CHILD_train.to_csv('large_CHILD_train.csv',index=False)

large_CHILD_test=app_test[app_test['CNT_CHILDREN']>=10]
large_CHILD_test.to_csv('large_CHILD_test.csv',index=False)
#孩子数 test 11个两次，20个1次。train >=11 8次 
#暂时不处理
'''


'''
abnormal_annuity_train=app_train[app_train['NAME_CONTRACT_TYPE']=='Revolving loans']
abnormal_annuity_test=app_test[app_test['NAME_CONTRACT_TYPE']=='Revolving loans' ]

abnormal_price_train=abnormal_annuity_train[ app_train['AMT_GOODS_PRICE']!=app_train['AMT_CREDIT'] ]
abnormal_price_test=abnormal_annuity_test[ app_test['AMT_GOODS_PRICE']!=app_test['AMT_CREDIT'] ]
abnormal_price_train.to_csv('abnormal_price_train.csv',index=False)
abnormal_price_test.to_csv('abnormal_price_test.csv',index=False)
#少数的revolving credit和price不相等。大部分都相等。所以 price中的nan用credit fill
'''

chosen_train=app_train['AMT_GOODS_PRICE'].isnull()
#no_price=app_train[chosen_train]
#no_price.to_csv('train_No price.csv',index=False)
app_train.loc[chosen_train,'AMT_GOODS_PRICE']=app_train.loc[chosen_train,'AMT_CREDIT']
#app_train[chosen_train].to_csv('train_No price_filled.csv',index=False)
#发现所有train没有price的都是revolving loans,相对于总revolving loans （29279） 很小
#同时test 都有price 并且revolving 本来就少




'''
# name_type_suite  occupation_type 很多nan
chosen_train=app_train['NAME_TYPE_SUITE'].isnull()
no_suite=app_train[chosen_train]
no_suite.to_csv('train_No suite.csv',index=False)
chosen_test=app_test['NAME_TYPE_SUITE'].isnull()
no_suite=app_test[chosen_test]
no_suite.to_csv('test_No suite.csv',index=False)
'''
#train name_income_type 有 maternity leave 但是 test没有
#target==1比例太高，delete
print(app_train.loc[app_train['NAME_INCOME_TYPE']=='Maternity leave','TARGET'])
app_train=app_train[app_train['NAME_INCOME_TYPE']!='Maternity leave']

#train NAME_FAMILY_STATUS 有 Unkown 但是 test没有
#target==0比例太高，delete
print(app_train.loc[app_train['NAME_FAMILY_STATUS']=='Unknown','TARGET'])
app_train=app_train[app_train['NAME_FAMILY_STATUS']!='Unknown']



#abnormal_annuity_train=abnormal_annuity_train[ app_train['AMT_CREDIT']!=20*app_train['AMT_ANNUITY'] ]
#abnormal_annuity_test=abnormal_annuity_test[ app_test['AMT_CREDIT']!=20*app_test['AMT_ANNUITY'] ]
#abnormal_annuity_train.to_csv('abnormal_annuity_train.csv',index=False)
#abnormal_annuity_test.to_csv('abnormal_annuity_test.csv',index=False)
#revolving annunity alway credit/20 or credit/10

'''
chosen_train_out=app_train['AMT_CREDIT']>2245500
train_credit_out=app_train[chosen_train_out]
train_credit_out.to_csv('train_credit_out.csv',index=False)

chosen_test_out=app_test['AMT_CREDIT']>2000000
test_credit_out=app_test[chosen_test_out]
test_credit_out.to_csv('test_credit_out.csv',index=False)

train_credit_out['AMT_CREDIT'].plot.hist();
test_credit_out['AMT_CREDIT'].plot.hist();
#train credit 有些别test最大的大很多。暂时没看出问题
'''

#app_train.to_csv('../input/application_train.csv',index=False)
#app_test.to_csv('../input/application_test.csv',index=False)
#没太大提升，也没太变坏
#v5 change from v4 clean until family status column
#5 overall  0.809567  0.759189



app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
plt.show()

'''
#365243 是由于已经退休 NAME_INCOME_TYPE == Pensioner or Unemployed
a=app_train[(app_train['NAME_INCOME_TYPE']=='Pensioner') & (app_train['DAYS_EMPLOYED']==365243)]
print(len(a))
#55352
print((app_train['NAME_INCOME_TYPE']=='Pensioner').sum())
#55362 10人士pensioner还工作。
print((app_train['DAYS_EMPLOYED']==365243).sum())
#55374 umeployed 也是36243
a=app_train[(app_train['NAME_INCOME_TYPE']=='Pensioner') | (app_train['DAYS_EMPLOYED']==365243)]
print(len(a))
#55384


a=app_test[(app_test['NAME_INCOME_TYPE']=='Pensioner') & (app_test['DAYS_EMPLOYED']==365243)]
print(len(a))
#9273
print((app_test['NAME_INCOME_TYPE']=='Pensioner').sum())
#9273 0人士pensioner还工作。
print((app_test['DAYS_EMPLOYED']==365243).sum())
#9274 umeployed 也是36243
a=app_test[(app_test['NAME_INCOME_TYPE']=='Pensioner') | (app_test['DAYS_EMPLOYED']==365243)]
print(len(a))
#9274
'''


'''
#pensioner id employed days 填 mean-2*std
#unemployed days 填 0
a=app_train.loc[app_train['DAYS_EMPLOYED']!=365243,'DAYS_EMPLOYED'].describe()
b=app_test.loc[app_test['DAYS_EMPLOYED']!=365243,'DAYS_EMPLOYED'].describe()

#pensioner id employed days 填 mean-2*std
train_number_to_fill=a['mean']-a['std']*3
test_number_to_fill=b['mean']-b['std']*3

#填 寿命-平均开始工作的日子
train_birth_employ_gap=a['mean']-app_train.loc[app_train['DAYS_EMPLOYED']!=365243,'DAYS_BIRTH'].mean()
test_birth_employ_gap=b['mean']-app_test.loc[app_test['DAYS_EMPLOYED']!=365243,'DAYS_BIRTH'].mean()
'''

app_train['flag_no_job']=(app_train['DAYS_EMPLOYED']==365243).astype(int)
app_test['flag_no_job']=(app_test['DAYS_EMPLOYED']==365243).astype(int)
'''
app_train.loc[(app_train['DAYS_EMPLOYED']==365243),'DAYS_EMPLOYED']=np.nan
app_test.loc[(app_test['DAYS_EMPLOYED']==365243),'DAYS_EMPLOYED']=np.nan

'''
#假设365243的都没换过工作。所以gap between birth 和employ小
train_gap=app_train[app_train['DAYS_EMPLOYED']!=365243]
train_gap['GAP']=app_train['DAYS_EMPLOYED']-app_train['DAYS_BIRTH']

test_gap=app_test[app_test['DAYS_EMPLOYED']!=365243]
test_gap['GAP']=app_test['DAYS_EMPLOYED']-app_test['DAYS_BIRTH']

aM=train_gap.loc[train_gap['CODE_GENDER']=='M','GAP'].describe()
aF=train_gap.loc[train_gap['CODE_GENDER']=='F','GAP'].describe()
bM=test_gap.loc[test_gap['CODE_GENDER']=='M','GAP'].describe()
bF=test_gap.loc[test_gap['CODE_GENDER']=='F','GAP'].describe()

train_birth_employ_gap_M=aM['mean']-aM['std']
train_birth_employ_gap_F=aF['mean']-aF['std']

test_birth_employ_gap_M=bM['mean']-bM['std']
test_birth_employ_gap_F=bF['mean']-bF['std']


app_train.loc[(app_train['NAME_INCOME_TYPE']=='Pensioner') & (app_train['DAYS_EMPLOYED']==365243) & (app_train['CODE_GENDER']=='M'),'DAYS_EMPLOYED']=app_train.loc[(app_train['NAME_INCOME_TYPE']=='Pensioner') & (app_train['DAYS_EMPLOYED']==365243) & (app_train['CODE_GENDER']=='M'),'DAYS_BIRTH']+train_birth_employ_gap_M
app_train.loc[(app_train['NAME_INCOME_TYPE']=='Pensioner') & (app_train['DAYS_EMPLOYED']==365243) & (app_train['CODE_GENDER']=='F'),'DAYS_EMPLOYED']=app_train.loc[(app_train['NAME_INCOME_TYPE']=='Pensioner') & (app_train['DAYS_EMPLOYED']==365243) & (app_train['CODE_GENDER']=='F'),'DAYS_BIRTH']+train_birth_employ_gap_F

app_test.loc[(app_test['NAME_INCOME_TYPE']=='Pensioner') & (app_test['DAYS_EMPLOYED']==365243) & (app_test['CODE_GENDER']=='M'),'DAYS_EMPLOYED']=app_test.loc[(app_test['NAME_INCOME_TYPE']=='Pensioner') & (app_test['DAYS_EMPLOYED']==365243) & (app_test['CODE_GENDER']=='M'),'DAYS_BIRTH']+test_birth_employ_gap_M
app_test.loc[(app_test['NAME_INCOME_TYPE']=='Pensioner') & (app_test['DAYS_EMPLOYED']==365243) & (app_test['CODE_GENDER']=='F'),'DAYS_EMPLOYED']=app_test.loc[(app_test['NAME_INCOME_TYPE']=='Pensioner') & (app_test['DAYS_EMPLOYED']==365243) & (app_test['CODE_GENDER']=='F'),'DAYS_BIRTH']+test_birth_employ_gap_F

app_train.loc[(app_train['DAYS_EMPLOYED']>0) ,'DAYS_EMPLOYED']=0
app_test.loc[(app_test['DAYS_EMPLOYED']>0),'DAYS_EMPLOYED']=0
#有的会超过0，剩下的unemployed 也大于0，上面的全改0了
#app_train.loc[(app_train['NAME_INCOME_TYPE']=='Unemployed') & (app_train['DAYS_EMPLOYED']==365243),'DAYS_EMPLOYED']=0
#app_test.loc[(app_test['NAME_INCOME_TYPE']=='Unemployed') & (app_test['DAYS_EMPLOYED']==365243),'DAYS_EMPLOYED']=0


app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment');
plt.show()

# domain_lgb 提升些
# overall  0.822867  0.766751



#app_train.loc[app_train['OWN_CAR_AGE']>=74,'OWN_CAR_AGE']=74
app_train.loc[app_train['OWN_CAR_AGE'].isnull(),'OWN_CAR_AGE']=100
app_test.loc[app_test['OWN_CAR_AGE'].isnull(),'OWN_CAR_AGE']=100


#没车的人车龄改为100， 提升一小点
# overall  0.808628  0.759160
# overall  0.816157  0.766851

app_train.loc[app_train['NAME_INCOME_TYPE']=='Pensioner','OCCUPATION_TYPE']='Pensioner'
app_train.loc[app_train['NAME_INCOME_TYPE']=='Unemployed','OCCUPATION_TYPE']='Unemployed'
app_test.loc[app_test['NAME_INCOME_TYPE']=='Pensioner','OCCUPATION_TYPE']='Pensioner'
app_test.loc[app_test['NAME_INCOME_TYPE']=='Unemployed','OCCUPATION_TYPE']='Unemployed'

#还是有occupation_type nan,没找到原因
#occupation_missing_train=app_train[app_train['OCCUPATION_TYPE'].isnull()]
#occupation_missing_test=app_test[app_test['OCCUPATION_TYPE'].isnull()]
#occupation_missing_train.to_csv('./occupation_missing_train.csv',index=False)
#occupation_missing_test.to_csv('./occupation_missing_test.csv',index=False)

#填部分occupation后对base提升，domain提升小
#overall  0.811749  0.759330
#overall  0.815780  0.766875

app_train['CNT_PARENTS_MEMBERS']=app_train['CNT_FAM_MEMBERS']-app_train['CNT_CHILDREN']
app_test['CNT_PARENTS_MEMBERS']=app_test['CNT_FAM_MEMBERS']-app_test['CNT_CHILDREN']
#加入家长数目 下降了
#overall  0.805610  0.759015
#overall  0.821146  0.766410

#REGION_RATING_CLIENT_W_CITY  1,2,3 test中有个-1，换成1
app_test.loc[app_test['REGION_RATING_CLIENT_W_CITY']==-1,'REGION_RATING_CLIENT_W_CITY']=1

#ORGANIZATION_TYPE 如果是XNA因为 NAME_INCOME_TYPE 是 Pensioner or Unemployed
app_train.loc[app_train['ORGANIZATION_TYPE']=='XNA','ORGANIZATION_TYPE']=app_train.loc[app_train['ORGANIZATION_TYPE']=='XNA','NAME_INCOME_TYPE']
app_test.loc[app_test['ORGANIZATION_TYPE']=='XNA','ORGANIZATION_TYPE']=app_test.loc[app_test['ORGANIZATION_TYPE']=='XNA','NAME_INCOME_TYPE']

#ext_source 加入mean
app_train['EXT_SOURCE_AVE']=app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
app_test['EXT_SOURCE_AVE']=app_test[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)

#overall  0.810061  0.758674
#overall  0.822656  0.766981


#app_train.to_csv('./source_ave_train.csv',index=False)
#app_test.to_csv('./source_ave_test.csv',index=False)

# train OBS_30_CNT_SOCIAL_CIRCLE 348 1个
# test OBS_30_CNT_SOCIAL_CIRCLE 300+3 个
#其他最大 47
# train DEF_30_CNT_SOCIAL_CIRCLE 34 1个
# test DEF_30_CNT_SOCIAL_CIRCLE 34 3个
#其他最大 8

app_train.loc[app_train['OBS_30_CNT_SOCIAL_CIRCLE']>50,'OBS_30_CNT_SOCIAL_CIRCLE']=50
app_test.loc[app_test['OBS_30_CNT_SOCIAL_CIRCLE']>50,'OBS_30_CNT_SOCIAL_CIRCLE']=50

app_train.loc[app_train['DEF_30_CNT_SOCIAL_CIRCLE']>10,'DEF_30_CNT_SOCIAL_CIRCLE']=10
app_test.loc[app_test['DEF_30_CNT_SOCIAL_CIRCLE']>10,'DEF_30_CNT_SOCIAL_CIRCLE']=10

app_train.loc[app_train['OBS_60_CNT_SOCIAL_CIRCLE']>50,'OBS_60_CNT_SOCIAL_CIRCLE']=50
app_test.loc[app_test['OBS_60_CNT_SOCIAL_CIRCLE']>50,'OBS_60_CNT_SOCIAL_CIRCLE']=50

app_train.loc[app_train['DEF_60_CNT_SOCIAL_CIRCLE']>10,'DEF_60_CNT_SOCIAL_CIRCLE']=10
app_test.loc[app_test['DEF_60_CNT_SOCIAL_CIRCLE']>10,'DEF_60_CNT_SOCIAL_CIRCLE']=10

#DAYS_LAST_PHONE_CHANGE
#train有一个nan，test没有
app_train=app_train.dropna(subset=['DAYS_LAST_PHONE_CHANGE'])
#qrt train 有个大数 删除
app_train=app_train[app_train['AMT_REQ_CREDIT_BUREAU_QRT']!=261]

app_train.to_csv('./train_cleaned.csv',index=False)
app_test.to_csv('./test_cleaned.csv',index=False)

#change from v2 employed day m/F
