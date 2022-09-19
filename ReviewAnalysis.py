import numpy as np
import sklearn
import pickle
import streamlit as st

save_cv = pickle.load(open('C:/Users/be1031/Downloads/count-Vectorizer.sav','rb'))
model = pickle.load(open('C:/Users/be1031/Downloads/Review_Classification.sav', 'rb'))

def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'Positive review'
    else:
        return 'Negative review'

def main():
    st.title('Review Sentiment analaysis')
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


