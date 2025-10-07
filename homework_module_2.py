import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
df = pd.read_csv(url)

# Preparing dataset
columns = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[columns]
print(df.dtypes)

# EDA
# Look at the fuel_efficiency_mpg variable. Does it have a long tail?
print(df['fuel_efficiency_mpg'].describe())

sns.histplot(df['fuel_efficiency_mpg'], bins=50, kde=True)
#plt.show()

fuel_effciency_values = np.log1p(df['fuel_efficiency_mpg'])
sns.histplot(fuel_effciency_values, bins=50, kde=True)
#plt.show()

# It does have a long tail.


# Question 1
# There's one column with missing values. What is it?
result = df.isnull().sum().loc[lambda x: x > 0]
print(result)

print(f"Q1 Answer: Column with missing values is {result.index[0]} with {result.values[0]} missing values")

# Question 2
# What's the median (50% percentile) for variable 'horsepower'?
print("Q2 Answer: Median value for variable 'horsepower' is: ", df['horsepower'].median())



# Prepare and split the dataset
# Shuffle the dataset (the filtered one you created above), use seed 42.

n = len(df)
print(f"The size of the dataset is {n}")

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df.iloc[n_train:]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]

idx = np.arange(n)

np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

print("Shuffled train: ", df_train.head())

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

print("Size of y_train ", len(y_train))

print(df_train.iloc[10])


# Question 3
# We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# For computing the mean, use the training only!
# Use the validation dataset to evaluate the models and compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?

mean_hp = df_train['horsepower'].mean()

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def prepare_X(df, fillna):
    df = df.copy()
    if fillna == 'mean':
        df['horsepower'] = df['horsepower'].fillna(mean_hp)
    elif fillna == 'zero':
        df['horsepower'] = df['horsepower'].fillna(0)
    else:
        raise ValueError("fillna must be 'mean' or 'zero'")
    return df.values

# root mean squared error function
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

X_train = prepare_X(df_train, fillna='mean')

w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color='red', bins=50, alpha=0.5)
sns.histplot(y_train, color='blue', bins=50, alpha=0.5)
#plt.show()

print("RMSE with mean:", round(rmse(y_train, y_pred), 2))

X_train = prepare_X(df_train, fillna='zero')
w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color='red', bins=50, alpha=0.5)
sns.histplot(y_train, color='blue', bins=50, alpha=0.5)
#plt.show()

print("RMSE with zero:", round(rmse(y_train, y_pred), 2))


# Using sklearn
X_train = prepare_X(df_train, fillna='mean')
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
rmse_mean = np.sqrt(mean_squared_error(y_train, y_pred))
print("RMSE with mean (sklearn):", round(rmse_mean, 2))

X_train = prepare_X(df_train, fillna='zero')
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
rmse_zero = np.sqrt(mean_squared_error(y_train, y_pred))
print("RMSE with zero (sklearn):", round(rmse_zero, 2))

print("Q3 Answer: Filling NAs with mean gives better RMSE:", round(rmse_mean, 2))

# Question 4
# Now let's train a regularized linear regression.
# For this question, fill the NAs with 0.
# Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
# Use RMSE to evaluate the model on the validation dataset.
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
# If multiple options give the same best RMSE, select the smallest r.

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = prepare_X(df_train, fillna='zero')
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val, fillna='zero')
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(f"RMSE with r={r}: ", round(score, 2))

print(f"Q4 Answer: Because all r values give the same RMSE, the smallest r=0 is the answer to Q4")

# Question 5
# We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.
# For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))

rmse_scores = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)

    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.fuel_efficiency_mpg.values
    y_val = df_val.fuel_efficiency_mpg.values

    del df_train['fuel_efficiency_mpg']
    del df_val['fuel_efficiency_mpg']
    del df_test['fuel_efficiency_mpg']

    X_train = prepare_X(df_train, fillna='zero')
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X(df_val, fillna='zero')
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    rmse_scores.append(score)
    print(f"Seed {seed}: RMSE {round(score, 2)}")


# Note: Standard deviation shows how different the values are.
# If it's low, then all values are approximately the same.
# If it's high, the values are different.
# If standard deviation of scores is low, then our model is stable.
std = np.std(rmse_scores)
print("Q5 Answer: Standard deviation of RMSE scores is: ", round(std, 3))

# Question 6
# Split the dataset like previously, use seed 9.
# Combine train and validation datasets.
# Fill the missing values with 0 and train a model with r=0.001.
# What's the RMSE on the test dataset?

idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

x_full_train = prepare_X(df_full_train, fillna='zero')
y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(x_full_train, y_full_train, r=0.001)
x_test = prepare_X(df_test, fillna='zero')
y_pred = w0 + x_test.dot(w)
score = rmse(y_test, y_pred)
print("Q6 Answer: RMSE on the test dataset is: ", score)