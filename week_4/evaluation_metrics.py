import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

print("Categorical columns:", categorical_columns)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod'
]

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
print("Predicted probabilities:", y_pred[:10])
churn_decision = (y_pred >= 0.5).astype(int)
print("Churn decisions:", churn_decision[:10])
print((y_val == churn_decision).mean())

# Accuracy and Dummy Model

print("How many customers we have", len(y_val))
print("How many decisions were correct", (y_val == churn_decision).sum())

from sklearn.metrics import accuracy_score
accuracy_score(y_val, churn_decision)

print("Accuracy:", accuracy_score(y_val, churn_decision))

thresholds = np.linspace(0, 1, 21)
print("Thresholds:", thresholds)

scores = []
for t in thresholds:
    #churn_decision = (y_pred >= t).astype(int)
    # score = (y_val == churn_decision).mean()
    score = accuracy_score(y_val, y_pred >= t)
    print(f'Threshold: {t:.2f}, Accuracy: {score:.3f}')
    scores.append(score)

plt.plot(thresholds, scores)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Decision Threshold')
plt.grid()
#plt.show()

from collections import Counter
print(Counter(y_pred >= 1.0))
print(Counter(y_val))
print(y_val.mean())
print(1- y_val.mean())

# Confusion table
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

t = 0.5
predicted_positive = (y_pred >= t)
predicted_negative = (y_pred < t)

tp = (predicted_positive & actual_positive).sum()
tn = (predicted_negative & actual_negative).sum()
fp = (predicted_positive & actual_negative).sum()
fn = (predicted_negative & actual_positive).sum()

print("TP", tp)
print("TN", tn)
print("FP", fp)
print("FN", fn)

confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

print("Confusion Matrix:\n", confusion_matrix)

print("Normalized Confusion Matrix:")
confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum()
print(confusion_matrix_normalized.round(2))

# 4.4 Precision and Recall

# Precision is fraction of posistive predictions that are correct

accuracy = (tp + tn) / (tp + tn + fp + fn)
print("Accuracy:", accuracy)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
print("Precision:", precision)
print("We will send promotions to", tp + fp, "customers")

# Recall is fraction of correctly identified positive examples
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print("Recall:", recall)

# 4.5 ROC Curves

# TPR & FPR
tpr = tp / (tp + fn)
print("TPR:", tpr)
print("TPR and Recall are the same:", tpr == recall)
fpr = fp / (fp + tn)
print("FPR:", fpr)

scores = []
thresholds = np.linspace(0, 1, 101)
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predicted_positive = (y_pred >= t)
    predicted_negative = (y_pred < t)

    tp = (predicted_positive & actual_positive).sum()
    tn = (predicted_negative & actual_negative).sum()

    fp = (predicted_positive & actual_negative).sum()
    fn = (predicted_negative & actual_positive).sum()

    scores.append((t,tp, fp, fn, tn))

df_scores = pd.DataFrame(scores, columns=['thresholds', 'tp', 'fp', 'fn', 'tn'])

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

print(df_scores[::10])

plt.plot(df_scores.thresholds, df_scores['tpr'], label='TPR')
plt.plot(df_scores.thresholds, df_scores['fpr'], label='FPR')
plt.legend()
#plt.show()

# Random model
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
print("Random predictions:", y_rand[:10].round(3))

print(((y_rand >= 0.5) == y_val).mean())

def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)
    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predicted_positive = (y_pred >= t)
        predicted_negative = (y_pred < t)

        tp = (predicted_positive & actual_positive).sum()
        tn = (predicted_negative & actual_negative).sum()

        fp = (predicted_positive & actual_negative).sum()
        fn = (predicted_negative & actual_positive).sum()

        scores.append((t,tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores, columns=['thresholds', 'tp', 'fp', 'fn', 'tn'])

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

df_rand = tpr_fpr_dataframe(y_val, y_rand)
print("Random model TPR/FPR:\n", df_rand[::10])

plt.plot(df_rand.thresholds, df_rand.tpr, label='TPR', linestyle='--')
plt.plot(df_rand.thresholds, df_rand.fpr, label='FPR', linestyle='--')
plt.legend()
plt.grid()
#plt.show()


# Ideal model
num_negative = (y_val == 0).sum()
num_positive = (y_val == 1).sum()
print("Number of negatives:", num_negative)
print("Number of positives:", num_positive)

y_ideal = np.repeat([0, 1], [num_negative, num_positive])
print("Ideal predictions:", y_ideal[:10])

y_ideal_pred = np.linspace(0, 1, len(y_val))
print(" ", 1 - y_val.mean())
print(" ", ((y_ideal_pred >= 0.726) == y_ideal).mean())


df_ideal =tpr_fpr_dataframe(y_ideal, y_ideal_pred)
print("Ideal model TPR/FPR:\n", df_ideal[::10])

plt.plot(df_ideal.thresholds, df_ideal.tpr, label='TPR', linestyle=':')
plt.plot(df_ideal.thresholds, df_ideal.fpr, label='FPR', linestyle=':')
plt.legend()
plt.grid()
#plt.show()

# reset plot

plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='model')
#plt.plot(df_rand.fpr, df_rand.tpr, label='random')
plt.plot([0,1], [0,1], label='random')
#plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')
plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.grid()
plt.title('ROC Curve')
#plt.show()


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

plt.plot(fpr, tpr, label='model')
plt.plot([0,1], [0,1], label='random')
plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.grid()
plt.title('ROC Curve Scikit-learn')
#plt.show()

# 4.6 ROC AUC Area Under the ROC Curve

from sklearn.metrics import auc

auc_score = auc(fpr, tpr)
print("AUC:", auc_score)
auc_model = auc(df_scores.fpr, df_scores.tpr)
print("AUC (model):", auc_model)
auc_ideal = auc(df_ideal.fpr, df_ideal.tpr)
print("AUC (ideal):", auc_ideal)

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(df_scores.fpr, df_scores.tpr)

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_val, y_pred)
print(auc(fpr, tpr))
print("ROC AUC Score:", roc_auc)

neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

import random

n = 100000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

print(success / n)

n = 50000
np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

print("Estimated AUC:", (pos[pos_ind] > neg[neg_ind]).mean())

# 4.7 K-Fold Cross-Validation

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    #model = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train(df_train, y_train, C=0.001)

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

y_pred = predict(df_val, dv, model)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

train_idx, val_idx = next(kfold.split(df_full_train))

from tqdm.auto import tqdm

n_splits = 5
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    scores = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_index, val_index in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_index]
        df_val = df_full_train.iloc[val_index]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)

        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print("AUC scores for each fold:", scores)
print(np.mean(scores), np.std(scores))

train_idx, val_idx = next(kfold.split(df_full_train))
print("Train indices:", train_idx)
print("Validation indices:", val_idx)
print(len(train_idx), len(val_idx))

df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print("Final AUC on test set:", auc)