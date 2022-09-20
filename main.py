import numpy as np
import sklearn
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
sen = "I hate the food and its taste is also not too good"
res = test_model(sen)
print(res)