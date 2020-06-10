#importing all the nedded libs
import nltk
import spacy
import numpy as np
import en_core_web_sm
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from flask import (Flask, render_template, jsonify, redirect, url_for, request, make_response)

app = Flask(__name__)

words_english = set(nltk.corpus.words.words())
correct_words_set = set(nltk.corpus.brown.words())
lematizator = en_core_web_sm.load()
model = np.load('pickles/persuasive_model.pkl',allow_pickle=True)
tv = np.load('pickles/persuasive_vectorizer.pkl',allow_pickle=True)
pca = np.load('pickles/persuasive_pca.pkl',allow_pickle=True)
scaleer = np.load('pickles/persuasive_scaleer.pkl',allow_pickle=True)
selector =  np.load('pickles/persuasive_selector.pkl',allow_pickle=True)

def get_prediction(essay_example, words_english, correct_words_set, 
                   lematizator, model, tv, pca, scaleer, selector):
    
    #counting first numerical features
    essay_sentences = nltk.sent_tokenize(essay_example)
    sentences_count = len(essay_sentences)
    essay_words= [word for word in nltk.word_tokenize(essay_example) if word not in punctuation]
    words_count =len(essay_words)
    unique_words_count = len(set(essay_words))
    
    #punctuation counter and misspelled words counter
    punctuation_counter=0
    misspelledwords_counter=0
    for word in nltk.word_tokenize(essay_example):
        if word.lower() not in punctuation:
            if word.lower() not in correct_words_set:
                misspelledwords_counter+=1
        else:
            punctuation_counter+=1
    punctuatioon_count = punctuation_counter
    spelling_errors = misspelledwords_counter
    
    
    #other numerical features and english words counter
    character_count = len(essay_example)
    words_per_sent_ratio = round(words_count/sentences_count)
    avg_sen_len = round(character_count/sentences_count)
    temp_english_word=0
    for word in essay_words:
        if word in words_english:
            temp_english_word+=1
    english_words=temp_english_word
    
    #stop words removal and counter
    temp_stop_words_count = 0
    temp_nsw = []
    for word in essay_words:
        if word not in STOP_WORDS:
            temp_nsw.append(word)
        else:
            temp_stop_words_count+=1
    essay_words = temp_nsw
    stopwords_count = temp_stop_words_count
    
    #parts of speach tagger and numerical features
    list_of_pos_features = {'LS':0, 'TO':0, 'VBN':0, 'WP':0, 'UH':0, 'VBG':0, 'JJ':0, 'VBZ':0, 
                            'VBP':0, 'NN':0, 'DT':0, 'PRP':0, 'WP$':0, 'NNPS':0, 'PRP$':0, 'WDT':0, 'RB':0, 'RBR':0, 
                            'RBS':0, 'VBD':0, 'IN':0, 'FW':0, 'RP':0, 'JJR':0, 'JJS':0, 'PDT':0, 'MD':0, 'VB':0, 'WRB':0, 
                            'NNP':0, 'EX':0, 'NNS':0, 'SYM':0, 'CC':0, 'CD':0, 'POS':0}
    temp_pos_list = np.unique(np.array(nltk.pos_tag(essay_words))[:,1], return_counts=True)
    pos_list = list(zip(temp_pos_list[0],temp_pos_list[1]))
    for i in pos_list:
        if i[0] in list_of_pos_features.keys():    
            list_of_pos_features[i[0]]=i[1]
    essay_words = ' '.join(essay_words)
    lemmatized_essay_words = [t.lemma_ for t in lematizator(essay_words)]
    essay_words = lemmatized_essay_words
    token_count = len(essay_words)
    unique_token_count = len(set(essay_words))
    essay_words = ' '.join(essay_words)
    
    all_features = np.array([sentences_count,words_count,unique_words_count,punctuatioon_count,spelling_errors,
                            character_count,words_per_sent_ratio,avg_sen_len,english_words,
                            stopwords_count,list_of_pos_features['LS'], list_of_pos_features['TO'], list_of_pos_features['VBN'], list_of_pos_features['WP'], list_of_pos_features['UH'], 
                            list_of_pos_features['VBG'], list_of_pos_features['JJ'], list_of_pos_features['VBZ'], 
                            list_of_pos_features['VBP'], list_of_pos_features['NN'], list_of_pos_features['DT'], 
                            list_of_pos_features['PRP'], list_of_pos_features['WP$'], list_of_pos_features['NNPS'], 
                            list_of_pos_features['PRP$'], list_of_pos_features['WDT'], list_of_pos_features['RB'], list_of_pos_features['RBR'], 
                            list_of_pos_features['RBS'], list_of_pos_features['VBD'], list_of_pos_features['IN'], list_of_pos_features['FW'], list_of_pos_features['RP'], 
                            list_of_pos_features['JJR'],list_of_pos_features['JJS'], list_of_pos_features['PDT'],
                            list_of_pos_features['MD'], list_of_pos_features['VB'], list_of_pos_features['WRB'], 
                            list_of_pos_features['NNP'], list_of_pos_features['EX'], list_of_pos_features['NNS'], list_of_pos_features['SYM'],
                            list_of_pos_features['CC'], list_of_pos_features['CD'], list_of_pos_features['POS'],token_count,unique_token_count]) 
    
    #transforming our essay
    selected_sel_k_best = selector.transform(all_features.reshape(1, -1))
    essay_words_vectorized = tv.transform([essay_words])
    numerical_scaled = scaleer.transform(selected_sel_k_best)
    essay_words_vectorized_reduced = pca.transform(essay_words_vectorized)
    X_reduced = np.concatenate((essay_words_vectorized_reduced,numerical_scaled),axis=1)
    prediction = model.predict(X_reduced[0].reshape(1,-1))[0]
    
    return prediction

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    essay_example = [x for x in request.form.values()][0].lower()
    
    prediction = get_prediction(essay_example, words_english, correct_words_set, lematizator, model, tv, pca, scaleer, selector)
    
    return render_template('index.html', prediction_text='{}'.format(prediction))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    essay_example=list(data.values())[0]
    prediction = get_prediction(essay_example, words_english, correct_words_set, lematizator, model, tv, pca, scaleer, selector)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)