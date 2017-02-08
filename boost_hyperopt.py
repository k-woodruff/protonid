from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing

import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import xgboost as xgb

def load_data():

    # use pandas to import csvs
    data_pb   = pd.read_csv('data/bnb/featuresana_bnbmc_august_p_primary.csv')
    data_pb   = data_pb[data_pb.mckinetic >= 0.04]
    data_m1   = pd.read_csv('data/bnb/featuresana_bnbmc_august_mu.csv')
    data_i1   = pd.read_csv('data/bnb/featuresana_bnbmc_august_pi.csv')
    data_e1   = pd.read_csv('data/bnb/featuresana_bnbmc_august_em.csv')
    data_call = pd.read_csv('data/corsika/featuresana_corsikait_august_train.csv')
    data_c1   = data_call[data_call.mcpdg != 2212]
    data_cp   = data_call[(data_call.mcpdg == 2212) & (data_call.mckinetic >= 0.04)]
    pframes   = [ data_pb, data_cp ]
    data_p1   = pd.concat(pframes)

    # pull out features we want to use now
    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdqdx','enddqdx','dqdxdiff','dqdxratio',
                     'totaldqdx','averagedqdx','cosmicscore','coscontscore',
                     'pidpida','pidchi','cfdistance']
    data_p = data_p1[feature_names]
    data_m = data_m1[feature_names]
    data_i = data_i1[feature_names]
    data_e = data_e1[feature_names]
    data_c = data_c1[feature_names]

    # make training array
    X0 = np.array(data_p)
    X1 = np.array(data_m)
    X2 = np.array(data_i)
    X3 = np.array(data_e)
    X4 = np.array(data_c)
    data  = np.vstack([X0,X1,X2,X3,X4])

    # make class labels
    y0 = np.zeros(len(X0))
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2))*2
    y3 = np.ones(len(X3))*3
    y4 = np.ones(len(X4))*4
    label  = np.hstack([y0,y1,y2,y3,y4])

    # make weights
    w0 = np.ones(len(y0))*1.
    w1 = np.ones(len(y1))*np.true_divide(len(y0),len(y1))
    w2 = np.ones(len(y2))*np.true_divide(len(y0),len(y2))
    w3 = np.ones(len(y3))*np.true_divide(len(y0),len(y3))
    w4 = np.ones(len(y4))*np.true_divide(len(y0),len(y4))
    weight = np.hstack([w0,w1,w2,w3,w4])

    return data,label,weight

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
             'n_estimators': hp.quniform('n_estimators', 200, 1000, 5),
             'eta': hp.quniform('eta', 0.025, 0.05, 0.005),
             #'max_depth': hp.quniform('max_depth', 1, 13, 1),
             'max_depth': hp.choice('max_depth',[4,5,6,7,8,9,10,11,12]),
             'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
             'subsample':  hp.quniform('subsample', 0.5, 1, 0.1),
             'gamma': hp.quniform('gamma', 0.0, 0.5, 0.05),
             'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'num_class': 5,
             'eval_metric': 'mlogloss',
             'objective': 'multi:softprob',
             'nthread': 7,
             'silent': 1
            }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best

X,y,w = load_data()
print "Splitting data into train and valid ...\n\n"
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w, test_size=0.2,random_state=1234)

trials = Trials()

optimize(trials)


