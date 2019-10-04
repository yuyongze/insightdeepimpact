# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:42:13 2019

@author: yuyon
"""

import pandas as pd
import numpy as np
def calculate_func(features, coef, call):
    if call == 'structural':
        feature_sub = features[:,:7]
        coef_sub = coef[:,:7]
    elif call == 'syntactic':
        feature_sub = features[:,7:19]
        coef_sub = coef[:,7:19]
    elif call == 'topic':
        feature_sub = features[:,19:29]
        coef_sub = coef[:,19:29]
    elif call == 'lexical':
        feature_sub = features[:,29:229]
        coef_sub = coef[:,29:229]
    elif call == 'content':
        feature_sub = features[:,229:]
        coef_sub = coef[:,229:]   
    return np.dot(feature_sub,coef_sub.T)

def get_bucket_score(features,coef,dropdown_content):
    result = {}
    for call in dropdown_content.keys():
        result[call] = calculate_func(features, coef, call).reshape(-1)
    return result

def get_bucket_result(all_score_df,feature_scaled,model,dropdown_content):
    
    test_score = get_bucket_score(feature_scaled,model.coef_,dropdown_content)
    test_score_df = pd.DataFrame(test_score)
    all_score_quantile= all_score_df.quantile([0.33,0.66])
    bucket = {}
    for col in all_score_df.columns:
        # put test score into low , medium, high
        if test_score_df[col][0]<all_score_quantile[col][0.33]:
            bucket[col]='low'
        elif test_score_df[col][0]<all_score_quantile[col][0.66]:
            bucket[col]='medium'
        else:
            bucket[col]='high'
    return pd.DataFrame([bucket])