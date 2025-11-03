import pandas as pd
import numpy as np
import typing as t

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
import xgboost as xgb

url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)

variable = 'fuel_efficiency_mpg'
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numeric_columns = list(df.select_dtypes(include=['int64', 'float64']).columns)


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

(df_train, y_train), (df_val, y_val), (df_test, y_test), df_full_train = split_data(df)

# Question 1

dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)

feature_names = dv.get_feature_names_out()
tree_rules = export_text(dt, feature_names=feature_names)
print(tree_rules)

root_feature_idx = dt.tree_.feature[0]
root_feature_name = feature_names[root_feature_idx]
print("Root split feature:", root_feature_name)

# Question 2

# root mean squared error function
def rmse(y_train, y_pred):
    se = (y_train - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
print(rmse(y_val, y_pred))

# Question 3

result = []
for n in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    result.append((n, rmse(y_val, y_pred).round(3)))

df = pd.DataFrame(result, columns=['n', 'rmse'])
print(df)

plt.plot(df.n, df.rmse)
plt.xlabel('Number of trees')
plt.ylabel('rmse')
plt.show()

# Question 4

result = []
max_depths = [10, 15, 20, 25]
for d in max_depths:
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        result.append((d, n, rmse(y_val, y_pred)))

df = pd.DataFrame(result, columns=['d', 'n', 'rmse'])
print(df)

# compute the mean per depth
df_mean = df.groupby('d')['rmse'].mean().sort_values()
print(df_mean.round(3))

# Question 5

rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
print("Feature importances:", importances)
feature_names = dv.get_feature_names_out()

fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(fi.head(20))

for col in ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]:
    print(col, fi[col])

# Question 6

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

features = dv.get_feature_names_out().tolist()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]

# eta tunning
scores = {}
eta_params = [0.3, 0.1]
results  = []

for eta in eta_params:
    xgb_params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,

        'objective': 'reg:squarederror',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }
    evals_result = {}
    model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=100, evals_result=evals_result, verbose_eval=5)

    val_rmses = evals_result['val']['rmse']
    best_rmse = min(val_rmses)
    best_round = val_rmses.index(best_rmse) + 1

    results.append((eta, best_rmse, best_round))

    key = f"eta={eta}"
    df_results = pd.DataFrame({
        'num_iter': range(1, len(evals_result['train']['rmse']) + 1),
        'train_rmse': evals_result['train']['rmse'],
        'val_rmse': evals_result['val']['rmse']
    })
    scores[key] = df_results

results_sorted = sorted(results, key=lambda x: x[1])

for eta, rmse_val, round_ in results_sorted:
    print(f"eta={eta}: best val RMSE={rmse_val:.4f} at round {round_}")

for key, df_scores in scores.items():
    plt.plot(df_scores['num_iter'], df_scores['val_rmse'], label=key)

plt.legend()
plt.show()