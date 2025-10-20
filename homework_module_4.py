import pandas as pd
import numpy as np
import typing as t

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.model_selection import KFold

url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
df = pd.read_csv(url)

variable = 'converted'
df[variable] = df[variable].astype(int)
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numeric_columns = list(df.select_dtypes(include=['int64', 'float64']).columns)
numeric_columns.remove(variable)


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # Fill missing values
    df[categorical_columns] = df[categorical_columns].fillna('NA')
    df[numeric_columns] = df[numeric_columns].fillna(0.0)
    return df


def split_data(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[variable].values
    y_val = df_val[variable].values
    y_test = df_test[variable].values

    del df_train[variable]
    del df_val[variable]
    del df_test[variable]

    df_full_train = df_full_train.reset_index(drop=True)

    return (df_train, y_train), (df_val, y_val), (df_test, y_test), df_full_train


def accuracy_with_feature(cols, df_train, df_val):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[cols].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[cols].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    converted_decision = (y_pred >= 0.5)

    accuracy_score = (y_val == converted_decision).mean()

    return accuracy_score, y_pred


print("Columns with missing values: ", df.isnull().sum().loc[lambda x: x > 0])
df = prep_data(df)
print("Columns with missing values: ", df.isnull().sum().loc[lambda x: x > 0])

(df_train, y_train), (df_val, y_val), (df_test, y_test), df_full_train = split_data(df)

accuracy_score, y_pred = accuracy_with_feature(categorical_columns + numeric_columns, df_train, df_val)

print("Accuracy with all features:", accuracy_score)

# Question 1

df_train_q1 = df_full_train.reset_index(drop=True)
y_train_q1 = df_train_q1[variable].values
X_train_q1 = df_train_q1[numeric_columns].copy()

auc_by_num = {}
for col in numeric_columns:
    score = X_train_q1[col].values
    auc = roc_auc_score(y_train_q1, score)
    if auc < 0.5:
        auc = roc_auc_score(y_train_q1, -score)
    auc_by_num[col] = auc

print("\n[Q1] Single-feature ROC AUC (train):")
for k, v in auc_by_num.items():
    print(f"{k}: {v:.3f}")

q1_best = max(auc_by_num.items(), key=lambda kv: kv[1])[0]
print(f"[Q1] Best single numerical feature by ROC AUC: {q1_best}")

# Question 2

dv = DictVectorizer(sparse=False)
categorical = categorical_columns
numerical = numeric_columns
features = categorical + numerical

train_dict = df_train[features].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[features].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
model.fit(X_train, y_train)

y_val_pred = model.predict_proba(X_val)[:, 1]
q2_auc = roc_auc_score(y_val, y_val_pred)

print(f"\n[Q2] Validation ROC AUC (C=1.0): {q2_auc:.3f}")

# Question 3

precisions = []
recalls = []
result = {}
thresholds = np.arange(0.0, 1.01, 0.01)
for t in thresholds:
    pred = (y_val_pred >= t).astype(int)

    tp = ((pred == 1) & (y_val == 1)).sum()
    tn = ((pred == 0) & (y_val == 0)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precisions.append(precision)
    recalls.append(recall)

    result["threshold={:.3f}".format(t)] = {
        "precision": precision,
        "recall": recall
    }

precisions = np.array(precisions)
recalls = np.array(recalls)

fig, ax = plt.subplots(figsize=(8, 4), dpi=160)

ax.plot(thresholds, precisions, label='Precision')
ax.plot(thresholds, recalls, label='Recall')

ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision and Recall vs. Threshold')
ax.set_xlim(0, 1)

ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax.tick_params(axis='x', which='major', labelrotation=45)
ax.grid(True, which='major', alpha=0.6)
ax.grid(True, which='minor', alpha=0.25)
ax.legend()
fig.tight_layout()
#plt.show()
#=> 0.640


# Question 4

f1_scores = []
thresholds = np.arange(0.0, 1.01, 0.01)
f1_score_result = {}
for t in thresholds:
    pred = (y_val_pred >= t).astype(int)

    tp = ((pred == 1) & (y_val == 1)).sum()
    tn = ((pred == 0) & (y_val == 0)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1_score)

    f1_score_result [f"threshold={t:.3f}"] = {
        "f1_score": f1_score
    }

for k, v in f1_score_result.items():
    print(f"[Q4] {k}: F1 score = {v['f1_score']:.3f}")

f1_scores = np.array(f1_scores)
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = f1_scores.max()
print(f"\n[Q4] Best threshold by F1 score: {best_threshold:.3f} with F1 score: {best_f1:.3f}")

# Question 5

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train(df_train, y_train, C=1.0)

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

y_pred = predict(df_val, dv, model)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

train_idx, val_idx = next(kfold.split(df_full_train))

from tqdm.auto import tqdm

n_splits = 5
scores = []
for C in tqdm([1.0]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_index, val_index in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_index]
        df_val = df_full_train.iloc[val_index]

        y_train = df_train[variable].values
        y_val = df_val[variable].values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)

        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print("AUC scores for each fold:", scores)
print(np.mean(scores), np.std(scores))

cv_scores = np.array(scores)
print(f"\n[Q5] 5-fold CV AUC scores: {cv_scores.round(3)}")
print(f"[Q5] Mean AUC: {cv_scores.mean():.3f}, Std: {cv_scores.std():.6f}")

# Question 6

n_splits = 5
scores = []
for C in tqdm([0.000001, 0.001, 1]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_index, val_index in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_index]
        df_val = df_full_train.iloc[val_index]

        y_train = df_train[variable].values
        y_val = df_val[variable].values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)

        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print("AUC scores for each fold:", scores)
print(np.mean(scores), np.std(scores))