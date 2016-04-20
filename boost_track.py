#!/usr/bin/python
# this is the example script to use xgboost to train
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data():

    Np  = 2493.*6.
    Nmu = 9.*26226.*6.
    Nbg = Np*7227.

    # use pandas to import csvs
    data_p = pd.read_csv('../data/new-v5/tracks/features_bnbp_544_contained.csv',delimiter=',')
    #data_b = pd.read_csv('../data/features_mcc7_muons_contained.csv',delimiter=',')
    data_d = pd.read_csv('../data/new-v5/tracks/features_corsikaMC_15_contained.csv',delimiter=',')
    #data_d = pd.read_csv('../data/trafcks/features_data_extunb_100_contained.csv',delimiter=',')
    # pull out features we want to use now
    feature_names = ['primary','ntrajpoints','startdqdx','enddqdx',
                     'dqdxdiff','dqdxratio','totaldqdx','averagedqdx',
                     'theta','phi','starty','startz','endy','endz','length',
                     'cosmicscore','coscontscore','pidpida']
    data_x1 = data_p[feature_names]
    #data_x2 = data_b[feature_names]
    data_x3 = data_d[feature_names]
    # make training array
    X1 = np.array(data_x1)[:1142,:]
    #X2 = np.array(data_x2)[:5491,:]
    X3 = np.array(data_x3)[:1142,:]
    data  = np.vstack([X1,X3])
    # make class labels
    y1 = np.ones(len(X1))
    #y2 = np.zeros(len(X2))
    y3 = np.zeros(len(X3))
    label  = np.hstack([y1,y3])
    # make weights
    w1 = np.ones(len(X1))*Np/len(X1)
    #w2 = np.ones(len(X2))*Nmu/len(X2)
    w3 = np.ones(len(X3))*Nbg/len(X3)
    #weight = np.hstack([w1,w2,w3])
    weight = np.ones(len(data))

    return data,label,weight

def run_cv(data,label,weight):

    # configure weights
    #weight = np.ones(len(data))
    #sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
    #sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

    # print weight statistics
    #print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wpos/sum_wneg))

    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight'] = 2.5
    #param['scale_pos_weight'] = 100.*sum_wpos/sum_wneg
    param['eta'] = 0.05
    param['max_depth'] = 9
    param['eval_metric'] = 'error'
    param['silent'] = 1
    param['nthread'] = 3

    # you can directly throw param in, though we want to watch multiple metrics here
    #plst = list(param.items())+[('eval_metric', 'falsepos')]
    plst = list(param.items())

    # boost 25 tres
    num_round = 50

    test_error    = []
    test_falsepos = []
    test_falseneg = []
    ypredvec      = []
    indexvec      = []

    # get folds
    skf = StratifiedKFold(label, 10)
    for i, (train, test) in enumerate(skf):
        #print train, test
        Xtrain = data[train]
        ytrain = label[train]
        wtrain = label[train]
        Xtest  = data[test]
        ytest  = label[test]
        wtest = label[test]
        # make dmatrices from xgboost
        dtrain = xgb.DMatrix( Xtrain, label=ytrain )
        dtest  = xgb.DMatrix( Xtest )
        #watchlist = [ (dtrain,'train') ]

        bst   = xgb.train(plst, dtrain, num_round)
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)
        ypredvec.append(ypred)
        indexvec.append(test)

    return test_error,test_falsepos,test_falseneg,ytest,ypredvec,indexvec,bst

def compute_stats(ytest,ypred):
  
    ydiff         = ypred - ytest
    fold_falsepos = float(len(np.where(ydiff > 0.75)[0]))/len(np.where(ytest == 0)[0])
    fold_falseneg = float(len(np.where(ydiff < -0.25)[0]))/len(np.where(ytest == 1)[0])
    fold_error    = float(len(np.where(ydiff > 0.75)[0])+len(np.where(ydiff < -0.25)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg






