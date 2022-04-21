# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:54:48 2022

@author: Manoj
"""

import pandas as pd
from pickle import dump
from pickle import load
from sklearn.naive_bayes import  ComplementNB
import re

import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer 




data=pd.read_csv('tweets.csv')
data.location.fillna("No Location" , inplace= True)
Real_len = data[data['target'] == 1].shape[0]
Not_len = data[data['target'] == 0].shape[0]
data_0 = data[data['target']==0]
data_1 = data[data['target']==1]
data_0_sampled = data_0.sample(data.target.value_counts()[1])
frames = [data_1,data_0_sampled]
data = pd.concat(frames)
ps = PorterStemmer() 

def preprocess(x):
         x=re.sub('[^a-zA-Z ]', ' ', x)
         x=x.lower()
         x=x.split()  
         x=[word  for word in x if word not in set(stopwords.words('english'))]  
         x=[ps.stem(word) for word in x] 
         x=" ".join(x)
         return x
    
data["text"]=data["text"].apply(preprocess)
    


X=data["text"]
y=data['target']

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(stop_words='english')


X = tv.fit_transform(X)
X=pd.DataFrame(X.todense(),columns=tv.get_feature_names())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3) 


from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score



Model = ComplementNB()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
y_prob = Model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred)) # gives summary of precision, recall, f1-score, support
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_prob) # gives false positive rates, true positive rates and thresholds between fpr and tpr
r = auc(fpr, tpr)
print('AUC:', r)

# save the model to disk
dump(Model, open('ComplimentNB.pkl', 'wb'))

# load the model from disk
loaded_model = load(open('ComplimentNB.pkl', 'rb'))
result = loaded_model.score(X_train, y_train)
print(result)
dump(tv, open('vectorizer49.pkl', 'wb'))

