import pandas as pd
import numpy as np

# Q1
print(pd.__version__)

# Q2
csv = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv')
print(csv.head())
print("Records:", len(csv))

# Q3

# Get distinct count of column 'fuel_type' from csv
print("Distinct fuel types:", csv['fuel_type'].nunique())

# Q4 How many columns have missing values?
print("Columns with missing values:", csv.isnull().any().sum())

# Q5 What's the maximum fuel efficiency of cars from Asia?
print("Max fuel efficiency of cars from Asia:", csv[csv['origin'] == 'Asia']['fuel_efficiency_mpg'].max())

# Q6
# 1. Find the median value of the column 'horsepower' in the dataset
print("Median horsepower:", csv['horsepower'].median())
# 2. Next, calculate the most frequent value of the column 'horsepower' in the dataset
print("Most frequent horsepower:", csv['horsepower'].mode()[0])
# 3. Use fillna method to fill the missing values in the 'horsepower' column with the most frequent value from the previous step
csv['horsepower'] = csv['horsepower'].fillna(csv['horsepower'].mode()[0])
# 4. Now calculate the medan value of the 'horsepower' column once again
print("Median horsepower after filling missing values:", csv['horsepower'].median())

# Q7
# 1. Select all the cars from Asia
asia_cars = csv[csv['origin'] == 'Asia']
# 2. Select only the columns vehicle_weight and model_year
asia_cars = asia_cars[['vehicle_weight', 'model_year']]
# 3. Select the first 7 values
asia_cars = asia_cars.head(7)
print("Asia cars data:\n", asia_cars)
# 4. Get the underlying NumPy array. Let's call it X
X = asia_cars.to_numpy()
print("X:", X)
# 5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX
XTX = X.T.dot(X)
print("XTX:", XTX)
# 6. Invert XTX
XTX_inv = np.linalg.inv(XTX)
print("Inverse of XTX:", XTX_inv)
# 7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
# 8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = XTX_inv.dot(X.T).dot(y)
print("Result w:", w)
# 9. What's the sum of all the elements of the result?
print("Sum of elements in w:", w.sum())