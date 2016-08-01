#!/usr/bin/python
# this is the example script to use xgboost to train
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def predict_data(fname):

    # use pandas to import csvs
    data = pd.read_csv(fname)

    # pull out features we want to use now
    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdqdx','enddqdx','dqdxdiff','dqdxratio',
                     'totaldqdx','averagedqdx','cosmicscore','coscontscore',
                     'pidpida','pidchi','cfdistance']
    d_feat = data[feature_names]

    dpred =  xgb.DMatrix(d_feat)

    # load in bdts
    #bdt_multi = xgb.Booster(model_file='bdts2/bdt_multi.bst')
    bdt_multi = xgb.Booster(model_file='multi_score1.bst')

    # score dataset
    m_preds  = bdt_multi.predict(dpred)
    max_pred = np.argmax(m_preds,axis=1)

    # save scores
    data['mscore_p']   = m_preds[:,0]
    data['mscore_mu']  = m_preds[:,1]
    data['mscore_pi']  = m_preds[:,2]
    data['mscore_em']  = m_preds[:,3]
    data['mscore_cos'] = m_preds[:,4]
    data['mscore_max'] = max_pred

    data.to_csv(fname)
        
    return data

def score_cut(data, pcut):
    
    # get passing samples
    data_pass = data[data.mscore_p > pcut]

    return data_pass
