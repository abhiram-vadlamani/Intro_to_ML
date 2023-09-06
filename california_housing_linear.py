import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import fetch_california_housing

# load the california dataset
housing = fetch_california_housing()
print(housing.keys())
print(housing.DESCR)

# conver the data into a pandas dataframe
california = pd.DataFrame(housing.data, columns=housing.feature_names)
california.head()
california.isnull().sum()
california['MEDV'] = housing.target

# define the input and target variables
features = ['AveBedrms','Latitude','Longitude']
X = pd.DataFrame(np.c_[california['AveBedrms'],
                          california['Latitude'],
                           california['Longitude']], columns = features)
Y = california['MEDV']

# divide the data into testing and training
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=6)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# run a linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

plt.plot(sorted(Y_test))
ind = np.argsort(Y_test)
plt.scatter(range(len(ind)),y_test_predict[ind],color="orange",marker = '.')
plt.show()
