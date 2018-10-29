from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing

import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import xgboost as xgb
import boost_train as bt

def score(params):

    print "Training with params: " 
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)
    model = xgb.train(params,dtrain,num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0],5))
    score = log_loss(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}

def optimize(trials):
    space = {
             'n_estimators': hp.quniform('n_estimators', 200, 1000, 50),
             'eta': hp.uniform('eta', 0.025, 0.05),
             #'max_depth': hp.quniform('max_depth', 1, 13, 1),
             'max_depth': hp.choice('max_depth',[4,5,6,7,8,9,10,11,12]),
             'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
             'subsample':  hp.uniform('subsample', 0.5, 1),
             'gamma': hp.uniform('gamma', 0.0, 0.5),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
             'num_class': 5,
             'eval_metric': 'mlogloss',
             'objective': 'multi:softprob',
             'nthread': 6,
             'silent': 1
            }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=100)

    print best

X,y,w = bt.load_data()
print "Splitting data into train and valid ...\n\n"
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w, test_size=0.2,random_state=1234)

trials = Trials()

optimize(trials)


