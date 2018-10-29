#!/usr/bin/python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def load_data():

    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdedx','dedxratio','trtotaldedx','traveragededx',
                     'cosmicscore','coscontscore']
    # use pandas to import csvs
    data        = pd.read_csv('data/bnbcosmic/trackfeatures_prodgenie_bnb_nu_cosmic_uboone_mcc8.7_cali_dev_pmtrack_truncated_train.csv',index_col=False)
    data        = data[data.length >= 0.5]
    print('Loaded BNB+Cosmic and removed short tracks')

    # if using separate files for bnb and cosmic
    '''
    frames = [data_bnb, data_cosmic]
    data   = pd.concat(frames)
    '''

    data['label'] = 4
    data['label'] = np.where(data['mcpdg'] == 2212, 0,data['label'])
    data['label'] = np.where((np.abs(data['mcpdg']) == 13)&(data['mcorigin'] == 1), 1,data['label'])
    data['label'] = np.where((np.abs(data['mcpdg']) == 211)&(data['mcorigin'] == 1), 2,data['label'])
    data['label'] = np.where(((np.abs(data['mcpdg']) == 11)|(data['mcpdg'] == 22))&(data['mcorigin'] == 1), 3,data['label'])

    label = np.array(data['label'])
    print('Made class label array')
    l0 = len(np.where(label == 0)[0])
    l1 = len(np.where(label == 1)[0])
    l2 = len(np.where(label == 2)[0])
    l3 = len(np.where(label == 3)[0])
    l4 = len(np.where(label == 4)[0])

    # make weights --- weight everything to smallest set
    protonweight = 1.
    minlen = np.min(np.array([l0,l1,l2,l3,l4])[np.nonzero(np.array([l0,l1,l2,l3,l4]))])

    weight = np.zeros(len(label))
    weight[np.where(label == 0)] = np.true_divide(minlen,l0)*protonweight
    weight[np.where(label == 1)] = np.true_divide(minlen,l1)
    weight[np.where(label == 2)] = np.true_divide(minlen,l2)
    weight[np.where(label == 3)] = np.true_divide(minlen,l3)
    weight[np.where(label == 4)] = np.true_divide(minlen,l4)
    print('Made weight array')

    #data = np.array(data[feature_names])

    return np.array(data[feature_names]),label,weight


def run_cv(data,label,weight):

    test_error    = []
    test_falsepos = []
    test_falseneg = []
    scores        = np.zeros((6,len(label)))

    # get folds
    skf = StratifiedKFold(label, 5, shuffle=True)
    for i, (train, test) in enumerate(skf):
        print 'On fold {}'.format(i)
        #print train, test
        Xtrain = data[train]
        ytrain = label[train]
        wtrain = weight[train]
        Xtest  = data[test]
        ytest  = label[test]
        wtest  = weight[test]

        bst = make_bdt(Xtrain, ytrain, wtrain)

        dtest  = xgb.DMatrix( Xtest )
        ypred = bst.predict(dtest)
        fold_error,fold_falsepos,fold_falseneg = compute_stats(ytest,ypred)
        test_error.append(fold_error)
        test_falsepos.append(fold_falsepos)
        test_falseneg.append(fold_falseneg)
        for i in range(5):
            scores[i,test] = ypred[:,i]
        scores[5,test] = ytest
        print 'fold test error: {}'.format(fold_error)
        print 'fold false positive: {}'.format(fold_falsepos)
        print 'fold false negative: {}'.format(fold_falseneg)

    return test_error,test_falsepos,test_falseneg,scores

def make_bdt(data,label,weight):

    # setup parameters for xgboost
    param = {}
    # original hyperparameters:
    param['objective']         = 'multi:softprob'
    param['eta']               = 0.05
    param['eval_metric']       = 'merror'
    param['silent']            = 1
    param['nthread']           = 6
    param['max_depth']         = 6
    param['colsample_bytree']  = 1.0
    param['subsample']         = 0.75
    param['num_class']         = 5

    # you can directly throw param in, though we want to watch multiple metrics here
    plst = list(param.items())+[('eval_metric', 'mlogloss')]

    # boost 300 trees
    num_round = 300
    
    # make dmatrices from xgboost
    dtrain = xgb.DMatrix( data, label=label, weight=weight )
    bst    = xgb.train(plst, dtrain, num_round)
        
    return bst

def compute_stats(ytest,ypred):
  
    yscore        = ypred[:,0]
    fold_falsepos = float(len(np.where((yscore > 0.5) & (ytest != 0))[0]))/len(np.where(ytest != 0)[0])
    fold_falseneg = float(len(np.where((yscore < 0.5) & (ytest == 0))[0]))/len(np.where(ytest == 0)[0])
    fold_error    = float(fold_falsepos*len(np.where(ytest != 0)[0]) + fold_falseneg*len(np.where(ytest == 0)[0]))/len(ytest)

    return fold_error,fold_falsepos,fold_falseneg

