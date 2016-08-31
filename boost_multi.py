#!/usr/bin/python
# this is the example script to use xgboost to train
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data():

    # use pandas to import csvs
    data_p1 = pd.read_csv('data/bnb/featuresana_bnbmc_july_p_primary.csv')
    data_p1 = data_p1[data_p1.mckinetic >= 0.04]
    data_m1 = pd.read_csv('data/bnb/featuresana_bnbmc_july_mu.csv')
    data_i1 = pd.read_csv('data/bnb/featuresana_bnbmc_july_pi.csv')
    data_e1 = pd.read_csv('data/bnb/featuresana_bnbmc_july_em.csv')
    #data_d1 = pd.read_csv('data/bnb/featuresana_bnbmc_july_k.csv')
    data_c1 = pd.read_csv('data/cosmic/featuresana_bnbext_6000_july.csv')

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

def run_cv(data,label,weight):

    # setup parameters for xgboost
    param = {}
    # cosmic data parameters
    param['objective'] = 'multi:softprob'
    param['eta']               = 0.025
    param['eval_metric']       = 'merror'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 4
    param['max_depth']         = 13
    param['gamma']             = 0.7
    param['colsample_bytree']  = 0.5
    param['subsample']         = 0.8
    #param['reg_alpha']         = 1e-5
    param['num_class']         = 5

    # you can directly throw param in, though we want to watch multiple metrics here
    plst = list(param.items())+[('eval_metric', 'mlogloss')]

    # boost 25 tres
    num_round = 855

    test_error    = []
    test_falsepos = []
    test_falseneg = []
    scores        = np.zeros((6,len(label)))

    # get folds
    skf = StratifiedKFold(label, 10, shuffle=True)
    for i, (train, test) in enumerate(skf):
        print 'On fold {}'.format(i)
        #print train, test
        Xtrain = data[train]
        ytrain = label[train]
        wtrain = label[train]
        Xtest  = data[test]
        ytest  = label[test]
        wtest  = label[test]
        # make dmatrices from xgboost
        dtrain = xgb.DMatrix( Xtrain, label=ytrain, weight=weight )
        dtest  = xgb.DMatrix( Xtest )

        bst   = xgb.train(plst, dtrain, num_round)
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)
        for i in range(5):
            scores[i,test] = ypred[:,i]
        scores[5,test] = ytest

    return test_error,test_falsepos,test_falseneg,scores

def compute_stats(ytest,ypred):
  
    yscore        = ypred[:,0]
    fold_falsepos = float(len(np.where((yscore > 0.75) & (ytest != 0))[0]))/len(np.where(ytest != 0)[0])
    fold_falseneg = float(len(np.where((yscore < 0.75) & (ytest == 0))[0]))/len(np.where(ytest == 0)[0])
    fold_error    = float(fold_falsepos*len(np.where(ytest != 0)[0]) + fold_falseneg*len(np.where(ytest == 0)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg

def make_bdt(data,label,weight):

    # setup parameters for xgboost
    param = {}
    # cosmic data parameters
    param['objective']         = 'multi:softprob'
    param['eta']               = 0.025
    param['eval_metric']       = 'merror'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 4
    param['max_depth']         = 13
    param['gamma']             = 0.7
    param['colsample_bytree']  = 0.5
    param['subsample']         = 0.8
    #param['reg_alpha']         = 1e-5
    param['num_class']         = 5

    # you can directly throw param in, though we want to watch multiple metrics here
    plst = list(param.items())+[('eval_metric', 'mlogloss')]

    # boost 25 tres
    num_round = 855
    
    # make dmatrices from xgboost
    dtrain = xgb.DMatrix( data, label=label, weight=weight )
    bst    = xgb.train(plst, dtrain, num_round)
        
    return bst
