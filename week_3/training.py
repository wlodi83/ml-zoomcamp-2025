import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data)

print(df.head())

print(df.head().T)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

print(df.head().T)

print(df.dtypes)

# print(df[tc.isnull()][['customerid', 'totalcharges']]) - checking rows with totalcharges as '-'

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

print(df.churn.head())

df.churn = (df.churn == 'yes').astype(int)

print(df.churn.head())


from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

print(len(df_full_train), len(df_test))

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

print(df_train.head())

df_full_train = df_full_train.reset_index(drop=True)
print(df_full_train.isnull().sum())

print(df_full_train.churn.value_counts(normalize=True))
print(df_full_train.churn.mean())

global_churn_date = df_full_train.churn.mean()
global_churn_rate = round(global_churn_date, 2)
print(global_churn_rate)

categorical_columns = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
print(categorical_columns)

print("numeric_columns")
numerical = list(df_full_train.dtypes[df_full_train.dtypes != 'object'].index)

# remove item from list
numerical.remove('seniorcitizen')
numerical.remove('churn')
print(numerical)

print(df_full_train.columns)

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
               'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
               'paymentmethod']

print(df_full_train[categorical].nunique())

print(df_full_train.head())

# female gender churn rate

churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()

print(churn_female)

# male gender churn rate

churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()

print(churn_male)

global_churn = df_full_train.churn.mean()

print(global_churn)

# partner churn rate

churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
print(churn_partner)

churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
print(churn_no_partner)

# the difference between global churn rate and partner churn rate
print(global_churn - churn_partner)
print(global_churn - churn_no_partner)

# risk ratio

print(churn_partner / global_churn)
print(churn_no_partner / global_churn)

# get the churn rate per gender, mean, diff and risk ratio

print(df_full_train.groupby('gender').churn.mean())
print(df_full_train.groupby('gender').churn.agg(['mean', 'count']))
df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn
df_group['risk'] = df_group['mean'] / global_churn
print(df_group)

from IPython.display import display

for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)


# feature importance: mutual information
from sklearn.metrics import mutual_info_score

print(mutual_info_score(df_full_train.churn, df_full_train.contract))
print(mutual_info_score(df_full_train.churn, df_full_train.gender))
print(mutual_info_score(df_full_train.churn, df_full_train.partner))

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
print(mi.sort_values(ascending=False))

# feature importance: correlation

# 3.9 One Hot Encoding

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
print(train_dict)
X_train = dv.fit_transform(train_dict)

print(X_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

print(X_val)

# 3.9 Logisitic Regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-5, 5, 51)
y = sigmoid(z)

plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function')
plt.grid()
#plt.show()

# def linear_regression(xi):
#     result = w0
#
#     for j in range(len(w)):
#         result += w[j] * xi[j]
#
#     return result

# def logistic_regression(xi):
#     score = w0
#
#     for j in range(len(w)):
#         score += w[j] * xi[j]
#
#     result = sigmoid(score)
#     return result

# 3.10 Training logistic regression with scikit-learn

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)
print(model.intercept_[0])
print(model.coef_[0].round(3))

print(model.predict(X_train))
print(model.predict_proba(X_train))
print(model.predict_proba(X_train)[:,1])
y_pred = model.predict_proba(X_val)[:,1]
churn_decision = (y_pred >= 0.5)
print(churn_decision)

print(df_val[churn_decision].customerid)

(y_val == churn_decision).mean()
churn_decision.astype(int)

# Testing if predictions are correct
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
print(df_pred)

df_pred['correct'] = df_pred.prediction == df_pred.actual
print(df_pred)

print(df_pred.correct.mean())
print((y_val == churn_decision).mean())
# model is correct 80% of the time

# 3.11 Model interpretation
print(dv.get_feature_names_out())
# coefficients
model.coef_[0].round(3)

a = [1,2,3,4]
b = 'abcd'

print(dict(zip(a,b)))

print(dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3))))

small = ['contract', 'tenure', 'monthlycharges']
print("Small dataset")
print(df_train[small].iloc[:10].to_dict(orient='records'))
dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
print("Small DV feature names")
print(dv_small.get_feature_names_out())

X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression(random_state=1)
model_small.fit(X_train_small, y_train)
print("Small model coefficients")
print(model_small.intercept_)
w0 = model_small.intercept_[0]

print(model_small.coef_[0])
w = model_small.coef_[0]
w.round(3)

print(dict(zip(dv_small.get_feature_names_out(), w.round(3))))

# test for one customer
print(w0)
print(w[0])
print(w[1])
print(w[2])
print(w[3])
print(w[4])
print(sigmoid(w0 + w[0]*1 + w[1]*0 + w[2]*0 + w[3]*50.0 + w[4]*5))

print("The raw score")
print(w0 + w[0]*1 + w[1]*0 + w[2]*0 + w[3]*50.0 + w[4]*5)
print("The probability")
print(sigmoid(w0 + w[0]*1 + w[1]*0 + w[2]*0 + w[3]*50.0 + w[4]*5))
print(sigmoid(-2.47 + 0.97 + 50 * 0.027 + 1 * (-0.036)))

# 3.12 Using the model

dict_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dict_full_train)

y_full_train = df_full_train.churn.values

model = LogisticRegression(random_state=1)
model.fit(X_full_train, y_full_train)
print(model.intercept_[0])
print(model.coef_[0].round(3))

dict_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dict_test)
y_pred = model.predict_proba(X_test)[:,1]

churn_decision = (y_pred >= 0.5)
accuracy = (churn_decision == y_test).mean()
print("Accuracy:", accuracy)

customer = dict_test[-1]
print("Customer 10 data:")
print(customer)

x_small = dv.transform([customer])
print(x_small)

print(model.predict_proba(x_small)[0,1])

print(y_test[-1])