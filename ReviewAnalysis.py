import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import pickle
save_cv = pickle.load(open('count-Vectorizer.sav','rb'))
model = pickle.load(open('Review_Classification.sav', 'rb'))

def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'Positive review'
    else:
        return 'Negative review'

def main():
    st.title('Review Sentiment analysis')
    sentence=st.text_input('Enter the sentence to analyse')
    result=''
    if st.button('Analyse'):
        result=test_model(sentence)
        if sentence=='':
            st.warning('Invalid Review',icon="⚠️")
            result="Enter proper review"
    st.success(result)


if __name__=='__main__':
    main()


