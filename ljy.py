import gc

import pandas as pd

paths = r'D:\ljy\data_format1\data_format1'
data = pd.read_csv(f'{paths}/user_log_format1.csv', dtype={'time_stamp': 'str'})
data1 = pd.read_csv(f'{paths}/user_info_format1.csv')
data2 = pd.read_csv(f'{paths}/train_format1.csv')
submission = pd.read_csv(f'{paths}/test_format1.csv')
data_train = pd.read_csv(r'D:\ljy\data_format2\data_format2\train_format2.csv')

data2['origin'] = 'train'
submission['origin'] = 'test'
matrix = pd.concat([data2, submission], ignore_index=True, sort=False)
matrix.drop(['prob'], axis=1, inplace=True)
matrix = matrix.merge(data1, on='user_id', how='left')
data.rename(columns={'seller_id': 'merchant_id'}, inplace=True)

data['user_id'] = data['user_id'].astype('int32')
data['merchant_id'] = data['merchant_id'].astype('int32')
data['item_id'] = data['item_id'].astype('int32')
data['cat_id'] = data['cat_id'].astype('int32')
data['brand_id'] = data['brand_id'].fillna(0)
data['brand_id'] = data['brand_id'].astype('int32')
data['time_stamp'] = pd.to_datetime(data['time_stamp'], format='%H%M')

matrix['age_range'].fillna(0, inplace=True)
matrix['gender'].fillna(2, inplace=True)
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'] = matrix['gender'].astype('int8')
matrix['label'] = matrix['label'].astype('str')
matrix['user_id'] = matrix['user_id'].astype('int32')
matrix['merchant_id'] = matrix['merchant_id'].astype('int32')

del data1, data2
gc.collect()

# 特征处理
groups = data.groupby(['user_id'])
temp = groups.size().reset_index().rename(columns={0: 'u1'})
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds / 3600
matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(
    columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')

groups = data.groupby(['merchant_id'])
temp = groups.size().reset_index().rename(columns={0: 'm1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'user_id': 'm2',
    'item_id': 'm3',
    'cat_id': 'm4',
    'brand_id': 'm5'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0: 'm6', 1: 'm7', 2: 'm8', 3: 'm9'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

temp = data_train[data_train['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'm10'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

groups = data.groupby(['user_id', 'merchant_id'])
temp = groups.size().reset_index().rename(columns={0: 'um1'})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'item_id': 'um2',
    'cat_id': 'um3',
    'brand_id': 'um4'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
    0: 'um5',
    1: 'um6',
    2: 'um7',
    3: 'um8'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['time_stamp'].agg([('frist', 'min'), ('last', 'max')]).reset_index()
temp['um9'] = (temp['last'] - temp['frist']).dt.seconds / 3600
temp.drop(['frist', 'last'], axis=1, inplace=True)
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')

matrix['r1'] = matrix['u9'] / matrix['u7']  # 用户购买点击比
matrix['r2'] = matrix['m8'] / matrix['m6']  # 商家购买点击比
matrix['r3'] = matrix['um7'] / matrix['um5']  # 不同用户不同商家购买点击比

matrix.fillna(0, inplace=True)

temp = pd.get_dummies(matrix['age_range'], prefix='age')
matrix = pd.concat([matrix, temp], axis=1)
temp = pd.get_dummies(matrix['gender'], prefix='g')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop(['age_range', 'gender'], axis=1, inplace=True)

# train、test-setdata
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

del temp, matrix
gc.collect()

# 导入分析库
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.3)

model = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42

)

model.fit(
    X_train,
    y_train,
    eval_metric='auc',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10
)

prob = model.predict_proba(test_data)
submission['prob'] = pd.Series(prob[:, 1])
submission.drop(['origin'], axis=1, inplace=True)
submission.to_csv(r'D:\ljy\sample_submission.csv', index=False)
