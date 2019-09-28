# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:27:37 2019

@author: yuyon
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#import plotly.graph_objects as go
import dash_table as dt
import pandas as pd
from feature_process import feature_generator
from similarity import get_top_n_table
import pickle

# dash style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# loading database
filename = './Models/model_logic_v3.pickle'
model = pickle.load(open(filename, 'rb'))

with open('./Models/StanderdScaler_fullset_v1.pickle','rb') as f:
    scaler = pickle.load(f)

review_df = pd.read_csv('./Data/moview_review_with_fullset_both_tag.csv',index_col='Unnamed: 0').reset_index(drop=True)
movie_info = pd.read_csv('./Data/movie_2018_10000.csv',index_col='Unnamed: 0')
feature_scaled_df = pd.read_csv('./Data/feature_fullset_positive_scale_20.csv',index_col='Unnamed: 0')

dropdown_list = ['structural','syntactic','topic','lexical','content']
dropdown_content = {'structural':'e.g. # of word,# of sentence,..',
                    'syntactic': "gramma statistics,e.g. freq of VERB, ADJ, ADV...",
                    'topic': "based on topic e.g. family, chateracters, fellings,..",
                    'lexical':"key words, eg. film, action...",
                    'content': "intergrated content analysis"
                    }



app.layout =html.Div([
    html.Div(children=[
        html.H1(children='Deep Impact',style={'text-align': 'left'}),
        html.Div(children='''
                 Improve Movie Discourse, Better!\n''',style={'font-weight':'bold'}),
                 
        html.Div(children='''
                 A web application to help you evaluate and improve the impact of movie reviews.
        '''),
        html.Div('',style={'padding': 10}),
        html.Div(dcc.Textarea(placeholder='Enter your review title',
    		value='',
    		style={'width': '80%','font-size': '14px'}, id='review_title')),
        
        html.Div('Rate the movie:'),
        
        html.Div(dcc.Slider(
        min=0,
        max=10,
        step=1,
        value=10,
        marks={i:'{}'.format(i) for i in range(11)
        }, id='category_weight'),style={'width':'60%','margin-left':'50px', 'margin-right':'auto'}),
        
        html.Div('',style={'padding': 10}),
        
        html.Div(dcc.Textarea(placeholder='Enter your review text',
    		value='',
    		style={'width': '80%','height':'200px','font-size': '14px'}, id='review_text'),  
        ),
    
    
        html.Div('',style={'padding': 10}),
        html.Button(id='submit', children='Evaluate Review'),
        
        html.Div('',style={'padding': 30})],
    style={'width':'1000px', 'margin-left':'auto', 'margin-right':'auto'}),
    
    html.Div(id="results",style={'width':'800px', 'margin-left':'auto', 'margin-right':'auto'}),
    html.Div(id="output-data-upload",children='',style={'width': '800px', 'margin-right': 'auto', 'margin-left': 'auto'}),
    
    html.Div(id='dropdown',children=[
            dcc.Dropdown(
                        id='similarity-key',
                        options=[{'label': i+' similarity'+' ('+ dropdown_content[i]+')', 'value': i} for i in dropdown_list],
                        value='content')],
                    style={'width':'800px', 'margin-left':'auto', 'margin-right':'auto'}),
    html.Div(id="rated_table",children='',style={'width':'800px', 'margin-left':'auto', 'margin-right':'auto'}),
    
    html.Div('',style={'padding': 30}),
    
    html.Div([html.H2('About:'),
	html.Div("This project was built by Yongze Yu at Insight Data Science \
		during the Autumn 2019 Boston session."),
	html.A("Slides", href='...', target="_blank"),
	html.Div(""),
	html.A("Source Code", href='...', target="_blank"),
        html.Div('', style={'padding':40})
	],style={'width': '800px', 'margin-left': 'auto','margin-right': 'auto', })

],style={'width': '800px', 'margin-right': 'auto', 'margin-left': 'auto','columnCount': 1})




@app.callback(
    [Output(component_id="output-data-upload",  component_property='children'),
     Output(component_id="rated_table",  component_property='children')],
    [Input(component_id='submit', component_property='n_clicks'),
     Input(component_id='similarity-key', component_property='value')],
    [State('review_text','value')])

def update_graph(n,rank_by,review_text):
    if n:
        
        # get score
        feature = pd.DataFrame([feature_generator(review_text)])
        score =  model.predict_proba(feature)[0][1]
        # load top_n table
        # get positive df
        feature_scaled_df_pos = feature_scaled_df
        
        scaled_test_feature = pd.DataFrame(scaler.transform(feature), columns=  feature.columns)      
        top_n_display = get_top_n_table(feature_scaled_df_pos,scaled_test_feature,review_df, movie_info,rank_by)
        
        
        
        #setting table
        table = dt.DataTable(
            columns=[{"name": i, "id": i} for i in top_n_display.columns],
            data=top_n_display.to_dict('rank'),
            style_cell = {'font_size': '13px', 'font_family':'sans-serif', 
             'maxWidth':'350px', 
            'whiteSpace':'normal','text-align': 'left'},
            style_cell_conditional=[
                                        {'if': {'column_id': 'movie_title'},
                                         'width': '15%'},
                                        {'if': {'column_id': 'review_rating'},
                                         'width': '10%'},
                                    ],
            css=[{'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
            )
            
        children = [html.H4("Your Score is "+str(round(score*100, 2))+" out of 100 (probability of top reviews*)"),
                    html.Div("* Top reviews mean top 20% ranking by helpfulness and up to 20 reviews each movie."),
                    html.Div('',style={'padding': 10}),
                    html.Div('Choose the tap below to get similar reviews which in top review pool:',style={})            
        ]
    else:
        children =''
        table =''
    return (children , table)



if __name__ == '__main__':
    app.run_server(debug=True)