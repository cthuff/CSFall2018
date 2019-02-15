# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_wine 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# load data directly from the library 
data = load_wine()
# make a Pandas dataframe from input features
df = pd.DataFrame(data.data, columns = data.feature_names) # add the Class of wine (rated as 0,1 &2) to the dataframe 
df['Class'] = pd.Series(data.target)

plt.figure() # new plot
df.hist()
plt.show()

plt.figure() # new plot
corMat = df.corr(method='pearson')
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
scatter_matrix(df)
plt.show()


array = df.values
X = array[:,0:13]
y = array[:,13]
df.shape

df.head(20)

"""
Normally at this stage we need to perform exploratory data analysis, look at the 
descriptive statistics, histograms and check out scatter plots and correlation
heatmap. This section has been skipped to save time (correlation and scatter plots
take a lot of time. You have seen these plots for Diabetes dataset before)
"""

"""
Preparing the Data
In this section we will divide our data into attributes and labels 
and will then divide the resultant data into training and test sets. 
By doing this we can train our algorithm on one set of data and then test it out 
on a completely different set of data that the algorithm hasn't seen yet. 
This provides you with a more accurate view of how your trained 
algorithm will actually perform. We now split the data
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
y_test = label_binarize(y_test, classes=[0])
 #   Bagging ensembles

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


# bagging classifier object instantiation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=1, random_state=42)
# fit the data to the bagging classifier
bag_clf.fit(X_train, y_train)
# predict the output for the test set
y_pred = bag_clf.predict(X_test)

# determine accuracy score for the bagging method
print("Bagging Method:")
print(accuracy_score(y_test, y_pred))

# determine the probability prediction for the test set
y_prob_bag = bag_clf.predict_proba(X_test)
# score is the same as the probabilities
y_score_bag = y_prob_bag[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_bag,tpr_bag, threshold_bag = roc_curve(y_test, y_score_bag)

# Standard Decision Tree Classifier

# now use a standard decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
# fit the data using the training set
tree_clf.fit(X_train, y_train)
# predict output with the test set
y_pred_tree = tree_clf.predict(X_test)
# determine accuracy score for the decision tree classifier
print("Standard Decision Tree:")
print(accuracy_score(y_test, y_pred))
# calculate decision tree classifier probabilities
y_prob_tree = tree_clf.predict_proba(X_test)
# score is the same as the probabilities
y_score_tree = y_prob_tree[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_tree,tpr_tree, threshold_tree = roc_curve(y_test, y_score_tree)



# Random Forests

from sklearn.ensemble import RandomForestClassifier

# random forest classifier object instantiation with default setting, i.e. max_depth is not set to anything
# the default setting overfit each tree to the training set, no limit set for max_depth
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
# fit the data using the training set
rnd_clf.fit(X_train, y_train)
# calculate random forest classifier probabilities
y_prob_rf = rnd_clf.predict_proba(X_test)
# predict output with the test set
y_pred_rf = rnd_clf.predict(X_test)
# determine accuracy score for the random forest classifier
print("Random Forest:")
print(accuracy_score(y_test,y_pred_rf))
# score is the same as the probabilities
y_score_rf = y_prob_rf[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_rf,tpr_rf, threshold_rf = roc_curve(y_test, y_score_rf)

################  RF2 ############################################
# random forest classifier object instantiation with max_depth=2
rnd_clf2 = RandomForestClassifier(n_estimators=500, max_depth = 2, n_jobs=-1, random_state=42)
# fit the data using the training set
rnd_clf2.fit(X_train, y_train)
# calculate random forest classifier probabilities
y_prob_rf2 = rnd_clf2.predict_proba(X_test)
# predict output with the test set
y_pred_rf2 = rnd_clf2.predict(X_test)
# determine accuracy score for the random forest classifier
print("Random Forest2:")
print(accuracy_score(y_test,y_pred_rf2))
# score is the same as the probabilities
y_score_rf2 = y_prob_rf2[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_rf2,tpr_rf2, threshold_rf2 = roc_curve(y_test, y_score_rf2)

################  RF3 ############################################
# random forest classifier object instantiation with max_depth=3
rnd_clf3 = RandomForestClassifier(n_estimators=250, max_depth = 3, n_jobs=-1, random_state=42)
# fit the data using the training set
rnd_clf3.fit(X_train, y_train)
# calculate random forest classifier probabilities
y_prob_rf3 = rnd_clf3.predict_proba(X_test)
# predict output with the test set
y_pred_rf3 = rnd_clf3.predict(X_test)
# determine accuracy score for the random forest classifier
print("Random Forest3:")
print(accuracy_score(y_test,y_pred_rf3))
# score is the same as the probabilities
y_score_rf3 = y_prob_rf3[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_rf3,tpr_rf3, threshold_rf3 = roc_curve(y_test, y_score_rf3)

################  RF4 ############################################
# random forest classifier object instantiation with max_depth=4
rnd_clf4 = RandomForestClassifier(n_estimators=800, max_depth = 4, n_jobs=-1, random_state=42)
# fit the data using the training set
rnd_clf4.fit(X_train, y_train)
# calculate random forest classifier probabilities
y_prob_rf4 = rnd_clf4.predict_proba(X_test)
# predict output with the test set
y_pred_rf4 = rnd_clf4.predict(X_test)
# determine accuracy score for the random forest classifier
print("Random Forest3:")
print(accuracy_score(y_test,y_pred_rf4))
# score is the same as the probabilities
y_score_rf4 = y_prob_rf4[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_rf4,tpr_rf4, threshold_rf4 = roc_curve(y_test, y_score_rf4)

# ADA Boost classifier

from sklearn.ensemble import AdaBoostClassifier

# Adaptive boosting classifier object instantiation
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
# fit the data using the training set
ada_clf.fit(X_train, y_train)
# predict output with the test set
y_pred_ada = ada_clf.predict(X_test)
# determine accuracy score for the adaptive boosting classifier
print("ADABoost:")
print(accuracy_score(y_test, y_pred_ada))
# calculate adaptive boosting classifier probabilities
y_prob_ada = ada_clf.predict_proba(X_test)
# score is the same as the probabilities
y_score_ada = y_prob_ada[:,1]
# calculate false postive, true postive rates and therehold for ROC
fpr_ada,tpr_ada, threshold_ada = roc_curve(y_test, y_score_ada)



# plotting section

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# plot bagging ROC
ax.plot(fpr_bag,tpr_bag, linewidth=2, label = 'Bagging')

# plot standard decision tree classifier ROC
ax.plot(fpr_tree, tpr_tree, linewidth=2, label = 'Decision Tree')

# plot random forest with default setting ROC 
ax.plot(fpr_rf,tpr_rf, linewidth=2, label = 'Random Forest')

# plot random forest with default setting ROC 
ax.plot(fpr_rf2,tpr_rf2, linewidth=2, label = 'RF2')

# plot random forest with default setting ROC 
ax.plot(fpr_rf3,tpr_rf3, linewidth=2, label = 'RF3')

# plot random forest with default setting ROC 
ax.plot(fpr_rf4,tpr_rf4, linewidth=2, label = 'RF4')

# plot ada boost ROC
ax.plot(fpr_ada,tpr_ada, linewidth=2, label = 'Ada Boost')
plt.title('Comparison of Classifier Performance, ROC')
plt.legend(loc="best")
plt.show()

