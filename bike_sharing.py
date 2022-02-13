#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 08:38:59 2022

@author: gajanan
"""

import pandas as pd
import numpy as np

train=pd.read_csv("train.csv",parse_dates=["datetime"])
test=pd.read_csv("test.csv",parse_dates=["datetime"])

train["year"]=train["datetime"].dt.year
train["month"]=train["datetime"].dt.month
train["day"]=train["datetime"].dt.day
train["wday"]=train["datetime"].dt.weekday
train["hour"]=train["datetime"].dt.hour

train['season'] = train['season'].astype('category')
dum_df = pd.get_dummies(train, drop_first=True)

test["year"]=test["datetime"].dt.year
test["month"]=test["datetime"].dt.month
test["day"]=test["datetime"].dt.day
test["wday"]=test["datetime"].dt.weekday
test["hour"]=test["datetime"].dt.hour

test['season'] = test['season'].astype('category')
dum_df_test = pd.get_dummies(test, drop_first=True)
dum_df_test.drop(columns=['datetime'],inplace=True)

dum_df_reg = dum_df.drop(columns=['datetime','casual','count'])

X = dum_df_reg.drop('registered',axis=1)
y_reg = dum_df_reg['registered']

############## grid search #####################
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
depth_range = np.arange(4,20,5)
minsplit_range = np.arange(4,20,5)
minleaf_range = [5,15,25]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)


clf = DecisionTreeRegressor(random_state=2021)

kfold = KFold(n_splits=5, random_state=2020,shuffle=True)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y_reg)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

bm_cv_reg = cv.best_estimator_

#Prediction On Casual
dum_df_cas = dum_df.drop(columns=['datetime','registered','count'])
X_cas = dum_df_cas.drop('casual',axis=1)
y_cas = dum_df_cas['casual'] 

depth_range = np.arange(4,16,4)
minsplit_range = np.arange(4,15,3)
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)


clf = DecisionTreeRegressor(random_state=2021)

kfold = KFold(n_splits=5, random_state=2020,shuffle=True)
cv2 = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')
cv2.fit(X_cas,y_cas)
# Best Parameters
print(cv2.best_params_)

print(cv2.best_score_)

bm_cv_cas = cv2.best_estimator_

#prediction On Above Two Prediction Models


y_pred_reg = bm_cv_reg.predict(dum_df_test).round()
y_pred_cas = bm_cv_cas.predict(dum_df_test).round()

y_pred_cnt = y_pred_reg+y_pred_cas 

#using Lasso
from sklearn.linear_model import Lasso

parameters_lasso = dict(alpha=np.linspace(5,10,15))
clf_lasso = Lasso(random_state=2021)
cv_las = GridSearchCV(clf_lasso, param_grid=parameters_lasso,
                  cv=kfold,scoring='r2')
cv_las.fit(X,y_reg)
print(cv_las.best_params_)
print(cv_las.best_score_)
best_est_reg=cv_las.best_estimator_
cv_las_cas = GridSearchCV(clf_lasso, param_grid=parameters_lasso,
                  cv=kfold,scoring='r2')
cv_las_cas.fit(X_cas,y_cas)
print(cv_las_cas.best_params_)
print(cv_las_cas.best_score_)
best_est_casual=cv_las_cas.best_estimator_

y_pred_reg_L = best_est_reg.predict(dum_df_test).round()
y_pred_cas_L = best_est_casual.predict(dum_df_test).round()

y_pred_cnt_L = y_pred_reg_L+y_pred_cas_L
test_submision = pd.read_csv("test.csv")
tst_dt = test['datetime']

y_pred_cnt_L[y_pred_cnt_L < 0] = 0


submit = pd.DataFrame({'datetime':tst_dt, 'count':y_pred_cnt_L}) 
submit.to_csv("submit_dt_lasso.csv",index=False)