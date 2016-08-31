#!/usr/bin/python
# this is the example script to use xgboost to train
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data():

    # use pandas to import csvs
    data_c1 = pd.read_csv('data/corsika/features_openCOSMIC_MC_AnalysisTrees_train.csv')
    data_p1 = data_c1[data_c1.MCpdgCode == 2212]
    data_b1 = data_c1[data_c1.MCpdgCode != 2212]

    # pull out features we want to use now
    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdqdx','enddqdx','dqdxdiff','dqdxratio',
                     'totaldqdx','averagedqdx','cosmicscore','coscontscore',
                     'pidpida','pidchi']
    data_p = data_p1[feature_names]
    data_b = data_b1[feature_names]

    # make training array
    X1 = np.array(data_p)
    X2 = np.array(data_b)
    data  = np.vstack([X1,X2])

    # make class labels
    y1 = np.ones(len(X1))
    y2 = np.zeros(len(X2))
    label  = np.hstack([y1,y2])

    # make weights
    w1 = np.ones(len(X1))
    w2 = np.ones(len(X2))*np.true_divide(len(X1),len(X2))
    weight = np.hstack([w1,w2])

    return data,label,weight

def run_cv(data,label,weight):

    # setup parameters for xgboost
    param = {}
    # cosmic data parameters
    param['objective'] = 'binary:logistic'
    param['eta']               = 0.025
    param['eval_metric']       = 'error'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 4
    param['max_depth']         = 13
    param['gamma']             = 0.7
    param['colsample_bytree']  = 0.5
    param['subsample']         = 0.8
    #param['reg_alpha']         = 1e-5

    plst = list(param.items())

    # boost 25 tres
    num_round = 500

    test_error    = []
    test_falsepos = []
    test_falseneg = []
    scores        = np.zeros((2,len(label)))

    # get folds
    skf = StratifiedKFold(label, 5, shuffle=True)
    for i, (train, test) in enumerate(skf):
        #print train, test
        print 'on fold {}'.format(i)
        Xtrain = data[train]
        ytrain = label[train]
        wtrain = weight[train]
        Xtest  = data[test]
        ytest  = label[test]
        wtest  = weight[test]
        # make dmatrices from xgboost
        dtrain = xgb.DMatrix( Xtrain, label=ytrain, weight=wtrain)
        dtest  = xgb.DMatrix( Xtest, label=ytest, weight=wtest )

        bst   = xgb.train(plst, dtrain, num_round)
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)
        scores[0,test] = ypred
        scores[1,test] = ytest
        print 'test error: {}'.format(fold_error)

    return test_error,test_falsepos,test_falseneg,scores

def compute_stats(ytest,ypred):
  
    ydiff         = ypred - ytest
    fold_falsepos = float(len(np.where(ydiff > 0.75)[0]))/len(np.where(ytest == 0)[0])
    fold_falseneg = float(len(np.where(ydiff < -0.25)[0]))/len(np.where(ytest == 1)[0])
    fold_error    = float(len(np.where(ydiff > 0.75)[0])+len(np.where(ydiff < -0.25)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg

def make_bdt(data,label,weight):

    # setup parameters for xgboost
    param = {}
    # cosmic data parameters
    param['objective'] = 'binary:logistic'
    param['eta']               = 0.025
    param['eval_metric']       = 'error'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 4
    param['max_depth']         = 13
    param['gamma']             = 0.7
    param['colsample_bytree']  = 0.5
    param['subsample']         = 0.8
    #param['reg_alpha']         = 1e-5

    plst = list(param.items())

    # boost 25 tres
    num_round = 500
    
    # make dmatrices from xgboost
    dtrain = xgb.DMatrix( data, label=label, weight=weight )
    bst    = xgb.train(plst, dtrain, num_round)
        
    return bst
