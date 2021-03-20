# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:05:33 2020

@author: HanhPham
"""

import numpy 
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pickle import load
from pickle import dump

filename = 'real_estate_dataset.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)

#Summarize data
print(dataset.head(20))
print(dataset.tail(20))

print(dataset.shape)
print(dataset.columns)

print(dataset.dtypes)
print(dataset.info())
print(dataset.isnull().sum())


print(dataset['CRIM'].unique())
print(dataset['ZN'].unique())
print(dataset['INDUS'].unique())
print(dataset['CHAS'].unique())
print(dataset['NOX'].unique())
print(dataset['RM'].unique())
print(dataset['AGE'].unique())
print(dataset['DIS'].unique())
print(dataset['RAD'].unique())
print(dataset['TAX'].unique())
print(dataset['PTRATIO'].unique())
print(dataset['B'].unique())
print(dataset['LSTAT'].unique())

set_option('precision',1)
print(dataset.describe())

print(dataset['CRIM'].quantile(0.999))


#Data visualizations
set_option('precision', 2)
print(dataset.corr(method='pearson'))

dataset.hist(sharex=False, sharey= False, xlabelsize=1, ylabelsize=1)
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex= False, sharey=False, fontsize=8)
scatter_matrix(dataset)
pyplot.show()

#correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax. matshow( dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ticks = numpy.arange(0,14,1)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

#Split-out validation dataset
X = dataset.drop(['MEDV'], axis=1)
Y = dataset['MEDV']

validation_size=0.20
seed= 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state= seed)

num_folds = 10
scoring='neg_mean_squared_error'



# GBM Alogorithm Tunning  ==> Choose boosting stages = 400 (the default is 100)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators= numpy.array([50, 100, 150, 200 , 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state= seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result= grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params= grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

#Finalize model witd scaled data
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators= 400)
model.fit(rescaledX, Y_train)

#Train data
predictions = model.predict(rescaledX)
print('MSE: %f' %  mean_squared_error(Y_train, predictions))
print('MAE: %f' % mean_absolute_error(Y_train, predictions))
print('R^2: %f' % r2_score(Y_train, predictions))
print('RMSE: %f' % numpy.sqrt(mean_squared_error(Y_train, predictions)))

#Plot train data
pyplot.scatter(Y_train, predictions)
pyplot.xlabel("Prices")
pyplot.ylabel("Predicted prices")
pyplot.title("Prices vs Predicted prices")
pyplot.show()

#Test data
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print('MSE: %f' %  mean_squared_error(Y_validation, predictions))
print('MAE: %f' % mean_absolute_error(Y_validation, predictions))
print('R^2: %f' % r2_score(Y_validation, predictions))
print('RMSE: %f' % numpy.sqrt(mean_squared_error(Y_validation, predictions)))

#Plot test data
pyplot.scatter(Y_validation, predictions)
pyplot.xlabel("Predicted")
pyplot.ylabel("Residuals")
pyplot.title("Predicted vs residuals")
pyplot.show()

#Save model for later use
filesave = 'final_model.sav'
dump(model, open(filesave, 'wb'))

#Load model
loaded_model = load(open(filesave, 'rb'))
loaded_model.fit(rescaledX, Y_train)
rescaledValidationX = scaler.transform(X_validation)
predictions = loaded_model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
result = loaded_model.fit(rescaledX, Y_train)
print(result)
