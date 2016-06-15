#!/usr/bin/python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

def predict_data(fname):

    # takes data csv, predicts proton scores and saves the csv with scores added

    # use pandas to import csvs
    data = pd.read_csv(fname)

    # pull out features we want to use now
    feature_names = ['nhits','length','starty','startz','endy','endz','theta','phi',
                     'distlenratio','startdqdx','enddqdx','dqdxdiff','dqdxratio',
                     'totaldqdx','averagedqdx','cosmicscore','coscontscore',
                     'pidpida','pidchi','cfdistance']
    d_feat = data[feature_names]

    # mmake into xgboost matrix
    dpred =  xgb.DMatrix(d_feat)

    # load in bdts
    bdt_cos  = xgb.Booster(model_file='bdts/bdt_cos.bst')
    bdt_pi   = xgb.Booster(model_file='bdts/bdt_pi.bst')
    bdt_mu   = xgb.Booster(model_file='bdts/bdt_mu.bst')
    bdt_em   = xgb.Booster(model_file='bdts/bdt_em.bst')

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

def score_cut(data, pcut_cos,pcut_pi,pcut_mu,pcut_em):
    
    # returns pandas dataframe with all tracks that pass score cuts

    # get passing samples
    data_pass = data[(data.score_cosmic > pcut_cos) & (data.score_piminus > pcut_pi) & (data.score_muminus > pcut_mu) & (data.score_electron > pcut_em)]

    return data_pass
