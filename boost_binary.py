#!/usr/bin/python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data(tracktype):

    # Load in csvs as pandas dataframes, select features, and make into numpy arrays
    # returns data, labels, and weights for input to cv and making bdts

    # use pandas to import csvs
    data_p1 = pd.read_csv('data/bnb/featuresana_mcc7_bnbMC_Nu_2_p_primary.csv')

    if tracktype == 'muon':
        data_b1 = pd.read_csv('data/bnb/featuresana_mcc7_bnbMC_Nu_2_mu.csv')
    elif tracktype == 'pion':
        data_b1 = pd.read_csv('data/bnb/featuresana_mcc7_bnbMC_Nu_2_pi.csv')
    elif tracktype == 'em':
        data_b1 = pd.read_csv('data/bnb/featuresana_mcc7_bnbMC_Nu_2_em.csv')
    elif tracktype == 'cosmic':
        data_b1 = pd.read_csv('data/cosmic/featuresana_bnbext_6000_contained.csv')
    else:
        print 'No track type selected... using cosmics'
        data_b1 = pd.read_csv('data/cosmic/featuresana_bnbext_6000_contained.csv')

    # pull out features we want to use now
    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdqdx','enddqdx','dqdxdiff','dqdxratio',
                     'totaldqdx','averagedqdx','cosmicscore','coscontscore',
                     'pidpida','pidchi','cfdistance']
    data_p = data_p1[feature_names]
    data_b = data_b1[feature_names]

    # make training array
    X1   = np.array(data_p)
    X2   = np.array(data_b)
    data = np.vstack([X1,X2])

    # make class labels
    y1 = np.ones(len(X1))
    y2 = np.zeros(len(X2))
    label  = np.hstack([y1,y2])

    # make weights
    weight = np.ones(len(data))

    return data,label,weight


def run_cv(data,label,weight):

    # run cross validation on training set to get stats

    # configure weights
    wp = len(np.where(label == 1)[0])
    wd = len(np.where(label == 0)[0])
    print 'Scale pos. weight: {}'.format(np.true_divide(wd,wp))

    # setup parameters for xgboost
    param = {}
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight']  = np.true_divide(wd,wp)
    param['eta']               = 0.05
    param['eval_metric']       = 'error'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 2
    param['max_depth']         = 10
    param['gamma']             = 0.0
    param['colsample_bytree']  = 0.8
    param['subsample']         = 0.9
    param['reg_alpha']         = 1e-5

    # boost 25 tres
    num_round = 300

    test_error    = []
    test_falsepos = []
    test_falseneg = []
    scores        = np.zeros((2,len(label)))

    # get folds
    skf = StratifiedKFold(label, 10, shuffle=True)
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

        bst   = xgb.train(param, dtrain, num_round)
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)
        scores[0,test] = ytest
        scores[1,test] = ypred

    return test_error,test_falsepos,test_falseneg,scores

def compute_stats(ytest,ypred):
  
    ydiff         = ypred - ytest
    fold_falsepos = float(len(np.where(ydiff > 0.75)[0]))/len(np.where(ytest == 0)[0])
    fold_falseneg = float(len(np.where(ydiff < -0.25)[0]))/len(np.where(ytest == 1)[0])
    fold_error    = float(len(np.where(ydiff > 0.75)[0])+len(np.where(ydiff < -0.25)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg

def make_bdt(data,label,weight):

    # create and save tree model

    # configure weights
    wp = len(np.where(label == 1)[0])
    wd = len(np.where(label == 0)[0])
    print 'Scale pos. weight: {}'.format(np.true_divide(wd,wp))

    # setup parameters for xgboost
    param = {}
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight']  = np.true_divide(wd,wp)
    param['eta']               = 0.05
    param['eval_metric']       = 'error'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 2
    param['max_depth']         = 10
    param['gamma']             = 0.0
    param['colsample_bytree']  = 0.8
    param['subsample']         = 0.9
    param['reg_alpha']         = 1e-5

    # boost 25 tres
    num_round = 300
    
    # make dmatrices from xgboost
    dtrain = xgb.DMatrix( data, label=label )
    bst    = xgb.train(plst, dtrain, num_round)
        
    return bst
