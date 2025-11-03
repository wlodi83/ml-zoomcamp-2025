import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Lesson 6.2: Data Preparation

data = 'https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CreditScoring.csv'
df = pd.read_csv(data)
print(df.head())
df.info()

# prep data

df.columns = df.columns.str.lower()

print(df.head())

print(df.status.value_counts())

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

print(df.head())

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)
print(df.head())

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)
print(df.head())

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(records_values)
print(df.head())

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)
print(df.head())

print(df.describe().round())

print(df.income.max())

for c in ['income', 'assets', 'debt']:
    df[c]= df[c].replace(to_replace=99999999, value=np.nan)

print(df.income.max())
print(df.head())

print(df.status.value_counts())

df = df[df.status != 'unk'].reset_index(drop=True)

print(df.status.value_counts())

# train test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(len(df_train), len(df_val), len(df_test))

y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status']
del df_val['status']
del df_test['status']

print(len(df_train), len(df_val), len(df_test))

# Lesson 6.3: Decision Trees

def assess_risk(client):
    if client['records'] == 'yes':
        if client['job'] == 'partime':
            return 'default'
        else:
            return 'ok'
    else:
        if client['assets'] > 6000:
            return 'ok'
        else:
            return 'default'

xi = df_train.iloc[0].to_dict()
print(xi)

print(assess_risk(xi))

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = dt.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)

print(f'ROC AUC: {auc}')

y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)

print(f'ROC AUC train: {auc}')# => 1 overfitting, memorizing the data but failing to generalize


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)

print(f'ROC AUC train: {auc}')

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print(f'ROC AUC: {auc}')

from sklearn.tree import export_text
tree_rules = export_text(dt, feature_names=dv.get_feature_names_out())
print(tree_rules)

# 6.4 - Decision Tree Learning Algorithm

data = [
    [8000, 'default'],
    [2000, 'default'],
    [0, 'default'],
    [5000, 'ok'],
    [5000, 'ok'],
    [4000, 'ok'],
    [9000, 'ok'],
    [3000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'status'])
print(df_example)

df_example = df_example.sort_values('assets')
print(df_example)

Ts = [0, 2000, 3000, 4000, 5000, 8000]

print("--------------------------------------------")

Ts = [4000]
for t in Ts:
    df_left = df_example[df_example['assets'] <= t]
    df_right = df_example[df_example['assets'] > t]

    # misclafication rate / impurity
    print(f"df_left for {t}: {df_left}")
    print(df_left.status.value_counts(normalize=True))
    print(f"df_right for {t}: {df_right}")
    print(df_right.status.value_counts(normalize=True))

# best T is 3000, impurity is 10%

# test data with additional feature like debt

data = [
    [8000, 3000, 'default'],
    [2000, 1000, 'default'],
    [0, 1000, 'default'],
    [5000, 1000, 'ok'],
    [5000, 1000, 'ok'],
    [4000, 1000, 'ok'],
    [9000, 500, 'ok'],
    [3000, 2000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'debt', 'status'])
print(df_example)

df_example = df_example.sort_values('debt')
print(df_example)

thresholds = {
    'assets': [0, 2000, 3000, 4000, 5000, 8000],
    'debt': [500, 1000, 2000, 3000]
}

Ts = [0, 500, 1000, 2000, 3000]

for feature, Ts in thresholds.items():
    print("#########################################################")
    print(f"Feature: {feature}")
    for T in Ts:
        print(f"Threshold: {T}")
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]
        print("df_left ----------------------------------")
        print(f"{df_left}")
        print(df_left.status.value_counts(normalize=True))
        print("df_right ---------------------------------")
        print(f"{df_right}")
        print(df_right.status.value_counts(normalize=True))
    print("#########################################################")


"""
Decision tree learning algorithm

- find the best split
- stop if max_depth is reached
- if left is sufficiently large and not pure:
    - repeat for left
- if right is sufficiently large and not pure:
    - repeat for right
"""

# Decision Trees Parameter Tuning
'''
selecting max_depth
selecting min_samples_leaf
'''
for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print(f"{d} => {auc}")

scores = []

for d in [4, 5, 6, 7, 10, 16, 20, None]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, s, auc))
        print(f"{d}, {s} => {auc}")

df_scores = pd.DataFrame(scores, columns=['d', 's', 'auc'])
print(df_scores)
print(df_scores.sort_values('auc', ascending=False))

df_scores_pivot = df_scores.pivot(index='s', columns=['d'], values=['auc'])
print(df_scores_pivot.round(3))

import matplotlib.pyplot as plt
import seaborn as sns
#sns.heatmap(df_scores_pivot, annot=True)
#plt.show()

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)


# 6. Ensembles and random forest
'''
Board of experts
Ensembling models
Random forest - ensembling decision trees
Tuning random forests
'''

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(auc)
print(rf.predict_proba(X_val[[0]]))

scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n, auc))

df = pd.DataFrame(scores, columns=['n', 'auc'])
print(df)

# plt.plot(df.n, df.auc)
# plt.xlabel('Number of trees')
# plt.ylabel('AUC')
# plt.show()

scores = []
for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d, n, auc))

df_scores = pd.DataFrame(scores, columns=['d','n', 'auc'])
print(df_scores)


for d in [5, 10, 15]:
    df_subset = df_scores[df_scores['d'] == d]
    #plt.plot(df_subset['n'], df_subset['auc'], label='max_depth=%d' % d)

#plt.legend()
#plt.show()

max_depth = 10
scores = []
for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=s, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((s, n, auc))

df_scores = pd.DataFrame(scores, columns=['s','n', 'auc'])
print(df_scores)

columns = ['s', 'n', 'auc']
df_scores = pd.DataFrame(df_scores, columns=columns)
print(df_scores.head())


colors = ['black', 'blue', 'orange', 'red', 'grey']
sample_leaf_values = [1, 3, 5, 10, 50]

for s, col in zip(sample_leaf_values, colors):
    df_subset = df_scores[df_scores['s'] == s]
    #plt.plot(df_subset['n'], df_subset['auc'], color=col, label='min_sample_leaf=%s' % s)

#plt.legend()
#plt.show()


min_samples_leaf = 3
rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=1)
rf.fit(X_train, y_train)


# 6.7 Gradient boosting and XGBoost
'''
- Gradient boosting vs random forest
- Installing XGBoost
- Training the first model
- Performance monitoring
- Parsing xgboost's monitoring output
'''

import xgboost as xgb

features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 4,
    'seed': 1,
    'verbosity': 1
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)
y_pred = model.predict(dval)
roc = roc_auc_score(y_val, y_pred)
print(roc)

watchlist = [(dtrain, 'train'), (dval, 'val')]
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 4,
    'seed': 1,
    'verbosity': 1
}
evals_result = {}
model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=20, evals_result=evals_result, verbose_eval=5)


df_results = pd.DataFrame({
    'num_iter': range(1, len(evals_result['train']['auc']) + 1),
    'train_auc': evals_result['train']['auc'],
    'val_auc': evals_result['val']['auc'],
})

print(df_results)

# plt.plot(df_results.num_iter, df_results.train_auc, label='train')
# plt.plot(df_results.num_iter, df_results.val_auc, label='val')
#
# plt.legend()
# plt.show()


# 6.8 XGboost parameter tunning

'''
Tunning the following parameters:
- eta
- max_depth
- min_child_weight
'''

# eta tunning
scores = {}
eta_params = [0.3, 1.0, 0.1, 0.05, 0.01]

for eta in eta_params:
    xgb_params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }
    evals_result = {}
    model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=200, evals_result=evals_result, verbose_eval=5)

    key = f"eta={eta}"
    df_results = pd.DataFrame({
        'num_iter': range(1, len(evals_result['train']['auc']) + 1),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    scores[key] = df_results

print(scores.keys())

for key, df_scores in scores.items():
    plt.plot(df_scores['num_iter'], df_scores['val_auc'], label=key)

plt.legend()
plt.show()

# max_depth tunning

scores = {}
max_depth_params = [6, 3, 4, 10]

for d in max_depth_params:
    xgb_params = {
        'eta': 0.1,
        'max_depth': d,
        'min_child_weight': 1,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }
    evals_result = {}
    model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=200, evals_result=evals_result, verbose_eval=5)

    key = f"max_depth={d}"
    df_results = pd.DataFrame({
        'num_iter': range(1, len(evals_result['train']['auc']) + 1),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    scores[key] = df_results

print(scores.keys())

del scores['max_depth=10']
for key, df_scores in scores.items():
    plt.plot(df_scores['num_iter'], df_scores['val_auc'], label=key)

plt.ylim(0.8, 0.84)
plt.legend()
plt.show()

# min_child_weight tunning

scores = {}
min_child_weight_params = [1, 10, 30]

for mc in min_child_weight_params:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': mc,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }
    evals_result = {}
    model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=200, evals_result=evals_result, verbose_eval=5)

    key = f"min_child_weight={mc}"
    df_results = pd.DataFrame({
        'num_iter': range(1, len(evals_result['train']['auc']) + 1),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    scores[key] = df_results

print(scores.keys())

for key, df_scores in scores.items():
    plt.plot(df_scores['num_iter'], df_scores['val_auc'], label=key)

plt.ylim(0.82, 0.85)
plt.legend()
plt.show()

# final model
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# 6.9 Select the final models

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:,1]
roc = roc_auc_score(y_val, y_pred)
print(roc)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:,1]
roc = roc_auc_score(y_val, y_pred)
print(roc)

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dtrain, num_boost_round=175)

y_pred = model.predict(dval)
roc = roc_auc_score(y_val, y_pred)
print(roc)

df_full_train = df_full_train.reset_index(drop=True)
print(df_full_train.head())

y_full_train = (df_full_train['status'] == 'default').astype(int).values
print(y_full_train)
del df_full_train['status']

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 30,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)

y_pred = model.predict(dtest)
roc = roc_auc_score(y_test, y_pred)
print(roc)
