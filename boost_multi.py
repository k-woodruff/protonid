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

def parameter_opt(data,label,weight):
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective']         = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight']  = 2.
    #param['scale_pos_weight'] = 100.*sum_wpos/sum_wneg
    param['eta']               = 0.05
    param['eval_metric']       = 'error'
    param['silent']            = 1
    param['nthread']           = 6
    param['min_child_weight']  = 4
    param['max_depth']         = 9
    param['gamma']             = 0.0
    param['colsample_bytree']  = 0.8
    param['subsample']         = 0.8
    #param['reg_alpha']         = 1e-5

    # you can directly throw param in, though we want to watch multiple metrics here
    #plst = list(param.items())+[('eval_metric', 'falsepos')]
    #plst = list(param.items())

    dtrain = xgb.DMatrix(data,label=label)

    # boost 25 tres
    num_round = 200

    '''
    scale_pos_weights = [0.5,0.75,1.25]
    for spw in scale_pos_weights:
        param['scale_pos_weight'] = spw
        plst = list(param.items())+[('eval_metric', 'falsepos')]
        results = xgb.cv(param,dtrain,num_boost_round=num_round,nfold=10,stratified=True)
        print 'scale_pos_weight: ',spw,', test-error-mean: ',np.array(results['test-error-mean'])[-1],', test-error-std: ',np.array(results['test-error-std'])[-1]

    return
    '''
    results = xgb.cv(param,dtrain,num_boost_round=num_round,nfold=10,stratified=True)
    return results


def run_cv(data,label,weight):

    # configure weights
    #weight = np.ones(len(data))
    #sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
    #sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

    # print weight statistics
    #print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wpos/sum_wneg))
    #wp = len(np.where(label == 1)[0])
    #wd = len(np.where(label == 0)[0])

    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    # cosmic data parameters
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    #param['scale_pos_weight'] = 3.*np.true_divide(wd,wp)
    #print 'Scale pos. weight: {}'.format(3.*np.true_divide(wd,wp))
    #param['scale_pos_weight'] = 100.*sum_wpos/sum_wneg
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
    #plst = list(param.items())

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

    # configure weights
    #weight = np.ones(len(data))
    #sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
    #sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

    # print weight statistics
    #print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wpos/sum_wneg))
    #wp = len(np.where(label == 1)[0])
    #wd = len(np.where(label == 0)[0])

    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    # cosmic data parameters
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    #param['scale_pos_weight'] = 3.*np.true_divide(wd,wp)
    #print 'Scale pos. weight: {}'.format(3.*np.true_divide(wd,wp))
    #param['scale_pos_weight'] = 100.*sum_wpos/sum_wneg
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
    #plst = list(param.items())

    # boost 25 tres
    num_round = 855
    
    # make dmatrices from xgboost
    dtrain = xgb.DMatrix( data, label=label, weight=weight )
    bst    = xgb.train(plst, dtrain, num_round)
        
    return bst
