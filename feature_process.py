# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:04:19 2019

@author: yuyon
"""

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from itertools import permutations 
from gensim.models.doc2vec import Doc2Vec
import pickle

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def preprocessing(text,stem=True,lemma=False, sentence = False):
    """
        Tokenization of the text with Stemming.abs
        
        arg:
            text: string,  input text 
            
        return:
            cleaned tonkenized text list
    """
    clean_text = []
    input_str = word_tokenize(text.lower())
    stemmer = PorterStemmer()
    if stem:
        for word in input_str:
            clean_text.append(stemmer.stem(word))
    lemmer = WordNetLemmatizer()
    if lemma:
        for word in input_str:
                clean_text.append(lemmer.lemmatize(word))

    if sentence:
        cleaned_sentence = " ".join(clean_text)
        return cleaned_sentence
    return clean_text


def metadata(text): # 7 features
    """
    Extract metadata from the review text,including:
    1. total number of paragranphs;
    2. total number of sentences;
    3. total number of words;
    4. average sentences in paragraph;
    5. average words in sentences;
    6. medium sentences in paragraph;
    7. medium words in sentences;
    """
    #get list of paragraphs
    para_list = [i for i in text.split('\n') if i]
    
    # No.1 feature
    num_para = len(para_list)
    
    sentence_count = []
    word_count = []
    
    for para in para_list:
        # calculate how many sentence in a paragraph
        sentence_list = nltk.sent_tokenize(text)
        sentence_count.append(len(sentence_list))
        
        # calculate how many words in a sentence
        for word in sentence_list:
            word_count.append(len(WhitespaceTokenizer().tokenize(word)))
            
    # No.2 feature
    num_sentence = np.sum(sentence_count)
    # No.3 feature
    num_word = np.sum(word_count) 
    # No.4 feature
    avg_sentence = np.mean(sentence_count)
    # No.5 feature
    avg_word = np.mean(word_count)
    # No.6 feature
    med_sentence = np.median(sentence_count)
    # No.5 feature
    med_word = np.median(word_count)
    return dict(zip(['num_para','num_sentence','num_word','avg_sentence','avg_word','med_sentence','med_word'],
               [num_para,num_sentence,num_word,avg_sentence,avg_word,med_sentence,med_word]))
    

def syntax_sturcture(text):  # 156 features
    """
    This function will use POS (part of speech) taging to identify the writing style of the text
    """
    cleaned_text = preprocessing(text)
    uni_tag_dict = dict.fromkeys({'ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X'},0)
    taged_text_list = nltk.pos_tag(cleaned_text,tagset='universal')
           
    # get a permutations from 
    bi_tag_list = list(permutations(list(uni_tag_dict),2))
    
    double_tag_list = set()
    for tag in uni_tag_dict.keys():
        double_tag_list.add(tag+'_'+tag)
    double_tag_dict = dict.fromkeys(double_tag_list,0) 
    # 'ADJ_ADJ', 'ADP_ADP'...
    
    bi_tag_dict = set()
    for bi_tag in bi_tag_list:
        bi_tag_dict.add(str(bi_tag[0])+'_'+str(bi_tag[1]))
    
    # {'ADJ_ADP', 'ADJ_CONJ', 'ADP_ADJ', 'ADP_CONJ', 'CONJ_ADJ', 'CONJ_ADP'...}

    bi_tag_dict = dict.fromkeys(bi_tag_dict,0)
    bi_tag_dict.update(double_tag_dict)
    
    for index in range(len(taged_text_list)):
        taged_text = taged_text_list[index][1]  # 'ADP'
        try:
            next_taged_text = taged_text_list[index+1][1] # 'ADJ'
        except:
            break
        
        uni_tag_dict[taged_text] +=1
        bi_tag = taged_text+'_'+next_taged_text  #'ADP_ADJ'
        bi_tag_dict[bi_tag] +=1
    
    tag_dict = {**uni_tag_dict,**bi_tag_dict}
    
    return tag_dict

def topic_generator(text):
    """
    """
    with open("./Models/Movie_review_LDA_model_v2.pickle", "rb") as f:
          lda_model = pickle.load(f)
    with open("./Models/Movie_reviewvectorizer_v2.pickle", "rb") as f:
          vectorizer = pickle.load(f)
    
    topic_feature_key = []
    topic_feature = lda_model.transform(vectorizer.transform([text]))[0]
    for i in range(len(topic_feature)):
        topic_feature_key.append('Topic_'+str(i))
    return dict(zip(topic_feature_key,topic_feature.tolist()))


def tfidf_generator(text):
    """Process for most common tfidf tokens [100 features]
    Args:
        review_df - df of raw review_text from csv
    Returns:
        
    """
    with open("./Models/tfidf_reviews_v2.pickle", "rb") as f:
              tfidf_model = pickle.load(f)
    with open("./Models/tfidf_bigram_reviews_v2.pickle", "rb") as f:
              tfidf_bigram_model = pickle.load(f)
    
    
    text = preprocessing(text,sentence = True)
    tfidf = list(tfidf_model.transform([text]).toarray()[0]) # sparse matrix
    tfidf_feature_names = tfidf_model.get_feature_names()

    tfidf_dict = dict(zip(tfidf_feature_names,tfidf))
    

    tfidf_bigram = list(tfidf_bigram_model.transform([text]).toarray()[0])
    tfidf_bigram_feature_names = tfidf_bigram_model.get_feature_names()
    
    tfidf_bigram_dict = dict(zip(tfidf_bigram_feature_names,tfidf_bigram))

    tfidf_feature_dict = {**tfidf_dict,**tfidf_bigram_dict}
    

    return tfidf_feature_dict



def doc2vec_generator(text):
    model = Doc2Vec.load('./Models/d2v_v2.model')
    content_feature = model.infer_vector(word_tokenize(text.lower()))
    
    column_name = []
    for i in range(100):
        column_name.append('doc2vec_'+str(i))
    content_feature_dict = dict(zip(column_name,content_feature))
    return content_feature_dict

def feature_generator(text):
    """
    return : all feature generated 
    """
    feature_dict = {}
    feature_dict.update(metadata(text))
    feature_dict.update(syntax_sturcture(text))
    feature_dict.update(topic_generator(text))
    feature_dict.update(tfidf_generator(text))
    feature_dict.update(doc2vec_generator(text))
    return feature_dict

if __name__ == '__main__':
    feature_generator('test here')