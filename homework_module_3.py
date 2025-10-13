import pandas as pd
import numpy as np


# In this homework, we will use the lead scoring dataset Bank Marketing dataset. Download it from here.
#
# Or you can do it with wget:
#
# wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv'
df = pd.read_csv(url)
print(df.head())

# In this dataset our desired target for classification task will be converted variable - has the client signed up to the platform or not.


#Data preparation
# Check if the missing values are presented in the features.
print("Columns with missing values: ", df.isnull().sum().loc[lambda x: x > 0])
# If there are missing values:
# For categorical features, replace them with 'NA'
categorical_features = df.select_dtypes(include=['object']).columns
print("Categorical Features:", categorical_features)
# For numerical features, replace with with 0.0
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Features:", numerical_features)
# Replace missing values
df[categorical_features] = df[categorical_features].fillna('NA')
df[numerical_features] = df[numerical_features].fillna(0.0)
print(df.head())
print("After filling missing values, any missing values left? ", df.isnull().sum().loc[lambda x: x > 0])


# Question 1
# What is the most frequent observation (mode) for the column industry?
#
# NA
# technology
# healthcare
# retail
print(df['industry'].value_counts(dropna=False))
print("Most frequent observation (mode) for the column industry is: ", df['industry'].mode(dropna=False)[0])

# Question 2. Biggest correlation (1 point)
#
# interaction_count and lead_score
# number_of_courses_viewed and lead_score
# number_of_courses_viewed and interaction_count
# annual_income and interaction_count

print(df.head().T)

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(len(df_full_train), len(df_test))
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(df_train.head())
print(df_val.head())
print(df_test.head())

y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values

del df_train['converted']
del df_val['converted']
del df_test['converted']

print(df_train.head())

df_full_train = df_full_train.reset_index(drop=True)
print(df_full_train.isnull().sum())

print(df_full_train.converted.value_counts(normalize=True))
print(df_full_train.converted.mean())

global_converted_rate = df_full_train.converted.mean()
global_converted_rate_rounded = round(global_converted_rate, 2)
print("Global converted rate (rounded): ", global_converted_rate_rounded)

categorical_features = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
print("Categorical features: ", categorical_features)
numerical_features = list(df_full_train.dtypes[df_full_train.dtypes != 'object'].index)
print("Numerical features: ", numerical_features)

numerical_features.remove('converted')

print(df_full_train.columns)

categorical = ['lead_source', 'industry', 'employment_status', 'location']

# Show the number of unique values for each categorical feature.
print(df_full_train[categorical].nunique())

# show unique values for each categorical feature
for col in categorical:
    print(f"Unique values for {col}: {df_full_train[col].unique()}")

print(df_full_train.head())

# Calculate the conversion rate for each value of the feature location.
converted_location_asia = df_full_train[df_full_train.location == 'asia'].converted.mean()
print("Converted rate for location 'asia': ", converted_location_asia)
converted_location_north_america = df_full_train[df_full_train.location == 'north_america'].converted.mean()
print("Converted rate for location 'north_america': ", converted_location_north_america)
converted_location_europe = df_full_train[df_full_train.location == 'europe'].converted.mean()
print("Converted rate for location 'europe': ", converted_location_europe)
converted_location_australia = df_full_train[df_full_train.location == 'australia'].converted.mean()
print("Converted rate for location 'australia': ", converted_location_australia)
converted_location_middle_east = df_full_train[df_full_train.location == 'middle_east'].converted.mean()
print("Converted rate for location 'middle_east': ", converted_location_middle_east)
converted_location_africa = df_full_train[df_full_train.location == 'africa'].converted.mean()
print("Converted rate for location 'africa': ", converted_location_africa)

print("----------------------------------------------------------------------")

# Calculate the conversion rate for each value of the feature industry.
converted_industry_NA = df_full_train[df_full_train.industry == 'NA'].converted.mean()
print("Converted rate for industry 'NA': ", converted_industry_NA)
converted_industry_education = df_full_train[df_full_train.industry == 'education'].converted
print("Converted rate for industry 'education': ", converted_industry_education.mean())
converted_industry_finance = df_full_train[df_full_train.industry == 'finance'].converted.mean()
print("Converted rate for industry 'finance': ", converted_industry_finance)
converted_industry_healthcare = df_full_train[df_full_train.industry == 'healthcare'].converted.mean()
print("Converted rate for industry 'healthcare': ", converted_industry_healthcare)
converted_industry_manufacturing = df_full_train[df_full_train.industry == 'manufacturing'].converted.mean()
print("Converted rate for industry 'manufacturing': ", converted_industry_manufacturing)
converted_industry_technology = df_full_train[df_full_train.industry == 'technology'].converted.mean()
print("Converted rate for industry 'technology': ", converted_industry_technology)

print("----------------------------------------------------------------------")

# Calculate the conversion rate for each value of the feature lead_source.
converted_lead_source_referral = df_full_train[df_full_train.lead_source == 'referral'].converted.mean()
print("Converted rate for lead_source 'referral': ", converted_lead_source_referral)
converted_lead_source_social_media = df_full_train[df_full_train.lead_source == 'social_media'].converted.mean()
print("Converted rate for lead_source 'social_media': ", converted_lead_source_social_media)
converted_lead_source_events = df_full_train[df_full_train.lead_source == 'events'].converted.mean()
print("Converted rate for lead_source 'social_media': ", converted_lead_source_events)
converted_lead_source_na = df_full_train[df_full_train.lead_source == 'NA'].converted.mean()
print("Converted rate for lead_source 'NA': ", converted_lead_source_na)
converted_lead_source_paid_ads = df_full_train[df_full_train.lead_source == 'paid_ads'].converted.mean()
print("Converted rate for lead_source 'paid_ads': ", converted_lead_source_paid_ads)
converted_lead_source_organic_search = df_full_train[df_full_train.lead_source == 'organic_search'].converted.mean()
print("Converted rate for lead_source 'organic_search: ", converted_lead_source_organic_search)

print("----------------------------------------------------------------------")

# Calculate the conversion rate for each value of the feature employment_status.
converted_employment_status_employed = df_full_train[df_full_train.employment_status == 'employed'].converted.mean()
print("Converted rate for employment_status 'employed': ", converted_employment_status_employed)
converted_employment_status_unemployed = df_full_train[df_full_train.employment_status == 'unemployed'].converted.mean()
print("Converted rate for employment_status 'unemployed': ", converted_employment_status_unemployed)
converted_employment_status_self_employed = df_full_train[df_full_train.employment_status == 'self_employed'].converted.mean()
print("Converted rate for employment_status 'self_employed': ", converted_employment_status_self_employed)
converted_employment_status_student = df_full_train[df_full_train.employment_status == 'student'].converted.mean()
print("Converted rate for employment_status 'student': ", converted_employment_status_student)
converted_employment_status_NA = df_full_train[df_full_train.employment_status == 'NA'].converted.mean()
print("Converted rate for employment_status 'NA': ", converted_employment_status_NA)

print("----------------------------------------------------------------------")

# Correlation between numerical features and target
print(df_full_train[numerical_features].corrwith(df_full_train.converted))
print("Correlation between annual_income and converted")
print(df_full_train[df_full_train.annual_income <= 50000].converted.mean())
print(df_full_train[df_full_train.annual_income > 50000].converted.mean())
print("Correlation between lead_score and converted")
print(df_full_train[df_full_train.lead_score < 0.2].converted.mean())
print(df_full_train[(df_full_train.lead_score >= 0.2) & (df_full_train.lead_score <= 1)].converted.mean())
print("Correlation between number_of_courses_viewed and converted")
print(df_full_train[df_full_train.number_of_courses_viewed < 2].converted.mean())
print(df_full_train[df_full_train.number_of_courses_viewed >= 2].converted.mean())
print("Correlation between interaction_count and converted")
print(df_full_train[df_full_train.interaction_count < 5].converted.mean())
print(df_full_train[df_full_train.interaction_count >= 5].converted.mean())

print("----------------------------------------------------------------------")

# Correlation between interaction_count and lead_score
print("Correlation between interaction_count and lead_score")
print(df_full_train[['interaction_count', 'lead_score']].corr().iloc[0, 1])
print(df_full_train.interaction_count.corr(df_full_train.lead_score))
print("Correlation between number_of_courses_viewed and lead_score")
print(df_full_train[['number_of_courses_viewed', 'lead_score']].corr().iloc[0, 1])
print(df_full_train.number_of_courses_viewed.corr(df_full_train.lead_score))
print("Correlation between number_of_courses_viewed and interaction_count")
print(df_full_train[['number_of_courses_viewed', 'interaction_count']].corr().iloc[0, 1])
print(df_full_train.number_of_courses_viewed.corr(df_full_train.interaction_count))
print("Correlation between annual_income and interaction_count")
print(df_full_train[['annual_income', 'interaction_count']].corr().iloc[0, 1])
print(df_full_train.annual_income.corr(df_full_train.interaction_count))

print("----------------------------------------------------------------------")

# Split the data
# Split your data in train/val/test sets with 60%/20%/20% distribution.
# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
# Make sure that the target value converted is not in your dataframe.

assert len(df_train) + len(df_val) + len(df_test) == len(df)
for X in (df_train, df_val, df_test):
    assert 'converted' not in X.columns

# Question 3
# Calculate the mutual information score between converted and other categorical variables in the dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).
# Which of these variables has the biggest mutual information score?
#
# industry
# location
# lead_source
# employment_status

from sklearn.metrics import mutual_info_score

print("Mutual information scores:")
print(mutual_info_score(df_full_train.converted, df_full_train.lead_source))
print(mutual_info_score(df_full_train.converted, df_full_train.industry))
print(mutual_info_score(df_full_train.converted, df_full_train.employment_status))
print(mutual_info_score(df_full_train.converted, df_full_train.location))

def mutual_info_converted_score(series):
    return mutual_info_score(series, df_full_train.converted)

mi = df_full_train[categorical].apply(mutual_info_converted_score)
print(mi.sort_values(ascending=False))

# Question 4
# Now let's train a logistic regression.
# Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
# What accuracy did you get?

print("----------------------------------------------------------------------")

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']
features = categorical + numerical

train_dict = df_train[features].to_dict(orient='records')
print(train_dict)
X_train = dv.fit_transform(train_dict)

print(X_train)

val_dict = df_val[features].to_dict(orient='records')
X_val = dv.transform(val_dict)

print(X_val)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

model.fit(X_train, y_train)

print("----------------------------------------------------------------------")

print(model.intercept_[0])
print(model.coef_[0].round(3))

print(model.predict(X_train))
print(model.predict_proba(X_train))
print(model.predict_proba(X_train)[:,1])
y_pred = model.predict_proba(X_val)[:,1]
print("Predicted probabilities for the validation set: ", y_pred)
converted_decision = (y_pred >= 0.5)
print(converted_decision)

(y_val == converted_decision).mean()
converted_decision.astype(int)

# Testing if predictions are correct
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = converted_decision.astype(int)
df_pred['actual'] = y_val
print(df_pred)

df_pred['correct'] = df_pred.prediction == df_pred.actual
print(df_pred)

print(df_pred.correct.mean())
print("acc =", (y_val == converted_decision).mean().round(3))

from sklearn.metrics import accuracy_score
print("acc sckit=", accuracy_score(y_val, converted_decision))


# Question 5
# Let's find the least useful feature using the feature elimination technique.
# Train a model using the same features and parameters as in Q4 (without rounding).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
# Which of following feature has the smallest difference?
#
# 'industry'
# 'employment_status'
# 'lead_score'
# Note: The difference doesn't have to be positive.

def accuracy_with_feature(cols):
    train_dict = df_train[cols].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[cols].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:,1]
    converted_decision = (y_pred >= 0.5)

    accuracy_score = (y_val == converted_decision).mean()

    return accuracy_score


baseline_acc = accuracy_with_feature(features)
print("Baseline accuracy: ", baseline_acc)

candidates = ['industry', 'employment_status', 'lead_score']

rows = []
for col in candidates:
    cols_wo = [c for c in features if c != col]
    acc_wo = accuracy_with_feature(cols_wo)
    diff = baseline_acc - acc_wo
    rows.append({'feature_dropped': col, 'baseline_acc': baseline_acc, 'acc_wo': acc_wo, 'diff': diff})

result = pd.DataFrame(rows).sort_values('diff', ascending=True)
print(result.reset_index(drop=True))


# Question 6
# Now let's train a regularized logistic regression.
# Let's try the following values of the parameter C: [0.01, 0.1, 1, 10, 100].
# Train models using all the features as in Q4.
# Calculate the accuracy on the validation dataset and round it to 3 decimal digits.

c_parameters = [0.01, 0.1, 1, 10, 100]

results = []

for c in c_parameters:
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:,1]
    converted_decision = (y_pred >= 0.5)

    acc = (y_val == converted_decision).mean()
    results.append({'C': c, 'accuracy': acc})

result = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print(result.reset_index(drop=True))