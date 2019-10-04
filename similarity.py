# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:36:19 2019

@author: yuyon
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def find_similarity(df,test,top_n = 5):
    """
        This funcion calcuate top n most silmuiar based on the feature selected
        arg:
            df: review database
        return:
            top_n_df: data from based on the similarity
        
    """
    similarity_matrix = cosine_similarity(df,np.reshape(np.array(test),(1,-1)))
    similarity_df = pd.DataFrame(similarity_matrix,columns=["similarity"],index=df.index)
    top_n_df = similarity_df.sort_values(by='similarity',ascending=False)[:top_n]
    return top_n_df

def create_similarity_table(df,test,review_df,movie_info):
    """
        Create table after find similar reviews.
    """
    similar_df = find_similarity(df,test)
    similar_df.index.rename('index',inplace=True)
    similar_df.reset_index(inplace=True)
    similar_df.index.rename('rank',inplace=True)

    movie_title = []
    review_rating = []
    review_text = []
    helpful_score = []
    for idx,row in similar_df.iterrows():
        index = row['index']
        # get movie title
        movie_title.append(movie_info[movie_info['ID'] == review_df.loc[index,'movie_id']]['title'].values[0])
        # get review rating
        review_rating.append(review_df.loc[index]['review_rating'])
        # get review text
        review_text.append(review_df.loc[index]['review_text'])

        # get reveiew helpful
        upvote = review_df.loc[index,'review_upvote']
        totalvote = review_df.loc[index,'review_totalvote']
        up_frac  = round(review_df.loc[index,'review_up_frac']*100,1)
        helpful_text = str(upvote)+'/'+str(totalvote)+' ('+str(up_frac)+'%)'
        helpful_score.append(helpful_text)
        
    similar_df['similarity'] = round(similar_df['similarity'],2)
    similar_df['movie_title'] =  movie_title
    similar_df['review_rating'] =  review_rating
    similar_df['review_text'] =  review_text
    similar_df['helpful_score'] =  helpful_score
    similar_df.drop('index',axis=1,inplace=True) 

    return similar_df

def get_top_n_table(scaled_feature_df,scaled_test,review_df,movie_info,order_by = 'content'):
    """
        Call this function to get similarity table by passing order_by different content.
    """
    structural =  scaled_feature_df.columns[:7]
    syntactic = scaled_feature_df.columns[7:19]
    topic = scaled_feature_df.columns[19:29]
    lexical = scaled_feature_df.columns[29:229]
    content = scaled_feature_df.columns[229:]
    
    if order_by == 'structural':
        df = scaled_feature_df[structural]
        test = scaled_test[structural]
    elif order_by == 'syntactic':
        df = scaled_feature_df[syntactic]
        test = scaled_test[syntactic]
    elif order_by == 'topic':
        df = scaled_feature_df[topic]
        test = scaled_test[topic]
    elif order_by == 'lexical':
        df = scaled_feature_df[lexical]
        test = scaled_test[lexical]
    elif order_by == 'content':
        df = scaled_feature_df[content]
        test = scaled_test[content]
        
    similar_df = create_similarity_table(df ,test,review_df,movie_info)
    return similar_df


if __name__ == '__main__':
    pass