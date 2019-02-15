# -*- coding: utf-8 -*-
"""
CS161 Homework 3 
@authors: jahan & Craig
Tested with Python 2.7
tested with Python 3.6 9/28/2018
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv
import numpy as np


filename = 'Baseball_salary.csv'
data = read_csv(filename)
data1 = data.drop(data.columns[19], axis=1) # when dataframe is read it generates a new index column, this will remove the extra column, check data in variable explorer

data1 = data1.drop(['League', 'Division', 'NewLeague'], axis = 1)
# separate features and response into two different arrays
array = np.log(data['Salary'].values)

# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = data1.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description = data1.describe()
print(description)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()
#
#
## correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
print(corMat)
## plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()
#
## scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()