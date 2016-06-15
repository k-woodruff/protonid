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
    bdt_cos  = xgb.Booster(model_file='bdts2/bdt_cos.py')
    bdt_pi   = xgb.Booster(model_file='bdts2/bdt_pi.py')
    bdt_mu   = xgb.Booster(model_file='bdts2/bdt_mu.py')
    bdt_em   = xgb.Booster(model_file='bdts2/bdt_em.py')

    # score dataset
    cos_pred = bdt_cos.predict(dpred)
    pi_pred  = bdt_pi.predict(dpred)
    mu_pred  = bdt_mu.predict(dpred)
    em_pred  = bdt_em.predict(dpred)

    # save scores
    data['score_cosmic']   = cos_pred
    data['score_piminus']  = pi_pred
    data['score_muminus']  = mu_pred
    data['score_electron'] = em_pred

    data.to_csv(fname)
        
    return data

def score_cut(data, pcut):
    
    # get passing samples
    data_pass = data[(data.score_cosmic > pcut) & (data.score_piminus > pcut) & (data.score_muminus > pcut) & (data.score_electron > pcut)]

    return data_pass
