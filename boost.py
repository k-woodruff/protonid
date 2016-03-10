#!/usr/bin/python
# this is the example script to use xgboost to train
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data():

    # use pandas to import csvs
    data_p = pd.read_csv('../data/features_mcc7_singlep.csv',delimiter=',')

    data_b = pd.read_csv('../data/features_mcc7_muons.csv',delimiter=',')
    data_d = pd.read_csv('../data/features_data_extunb_short.csv',delimiter=',')
    # pull out features we want to use now
    feature_names = ['primary','ndaughters','chargeinthit','startchargetotal',
            'endchargetotal','widthaverage','nhits','nhits0','nhits1','nhits2',
            'len','startx','starty','startz','endx','endy','endz']
    data_x1 = data_p[feature_names]
    data_x2 = data_b[feature_names]
    data_x3 = data_d[feature_names]
    # make training array
    X1 = np.array(data_x1)[:2500,:]
    X2 = np.array(data_x2)[:2500,:]
    X3 = np.array(data_x3)[:2500,:]
    data  = np.vstack([X1,X2,X3])
    # make class labels
    y1 = np.ones(len(X1))
    y2 = np.zeros(len(X2))
    y3 = np.zeros(len(X3))
    label  = np.hstack([y1,y2,y3])

    return data,label

def run_cv(data,label):

    # configure weights
    weight = np.ones(len(data))
    sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
    sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

    # print weight statistics
    print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight'] = 0.25*sum_wneg/sum_wpos
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['eval_metric'] = 'falsepos'
    param['silent'] = 1

    # you can directly throw param in, though we want to watch multiple metrics here
    plst = list(param.items())+[('eval_metric', 'error')]

    # boost 25 tres
    num_round = 50

    test_error    = []
    test_falsepos = []
    test_falseneg = []

    # get folds
    skf = StratifiedKFold(label, 10)
    for i, (train, test) in enumerate(skf):
        #print train, test
        Xtrain = data[train]
        ytrain = label[train]
        Xtest  = data[test]
        ytest  = label[test]
        # make dmatrices from xgboost
        dtrain = xgb.DMatrix( Xtrain, label=ytrain)
        dtest  = xgb.DMatrix( Xtest)
        #watchlist = [ (dtrain,'train') ]

        bst   = xgb.train(plst, dtrain, num_round)
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)

    return test_error,test_falsepos,test_falseneg

def compute_stats(ytest,ypred):
  
    ydiff         = ypred - ytest
    fold_falsepos = float(len(np.where(ydiff > 0.5)[0]))/len(np.where(ytest == 0)[0])
    fold_falseneg = float(len(np.where(ydiff < -0.5)[0]))/len(np.where(ytest == 1)[0])
    fold_error    = float(len(np.where(abs(ydiff) > 0.5)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg






