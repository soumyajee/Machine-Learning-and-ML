# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:49:10 2022

@author: Manoj
"""

import pandas as pd
import streamlit as st
st.title('Disastrous Tweet Classificaion')
#markdown
st.image('images.png')
st.markdown('This application is all about tweet sentiment analysis of disaster. We can analyse reviews of the disasters using this streamlit app.')
#sidebar
st.sidebar.title("About")
#st.sidebar.title('Sentiment analysis of disasters')
# sidebar markdown 
st.sidebar.markdown("🛫We can analyse disaster review from this application.🛫")
st.sidebar.markdown('Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programmatically monitoring Twitter (i.e. disaster relief organizations and news agencies). But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example: The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine. In this, we’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which ones aren’t.')
#loading the data (the csv file is in the same folder)
import joblib
import re
from pickle import dump
from pickle import load
import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
all_stopwords= stopwords.words('english')
all_stopwords.remove('not')

def main():
    form = st.form(key='sentiment-form')
    user_input = form.text_area('Enter the tweet to check ')
    #user_input = 'My name is Manoj'
    

    submit = form.form_submit_button('Submit')
    
    
    
    #n=len(user_input)
    
   
        
    if submit:
       corpus=[]
       #n = len(user_input)
       for i in range(0,1):
            sent = re.sub('[^a-zA-Z]', ' ',str(user_input))
            sent = sent.lower()
            sent = sent.split()
            sent = [ps.stem(word) for word in sent if not word in set(all_stopwords)]
            sent = ' '.join(sent)
            corpus.append(sent)
       
       loaded_model = load(open('ComplimentNB.pkl', "rb"))
        
       cv=load(open('vectorizer49.pkl','rb'))
       vect = cv.transform(corpus).toarray()
       #st.write(vect)
       X=pd.DataFrame(vect,columns=cv.get_feature_names())
       #st.write(X) 
       prediction = loaded_model.predict(X) 
       #st.write(prediction)
       prediction_proba = loaded_model.predict_proba(X)
       #st.write(prediction_proba)
        
       if prediction == [0]:
            st.success(f'Not Disastrous Tweet ')
       else:
            st.error(f'Disastrous Tweet  ')    
            
if __name__ == '__main__':
    main()            