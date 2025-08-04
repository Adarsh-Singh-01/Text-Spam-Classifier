import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string

@st.cache_data
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

@st.cache_resource
def load_and_train():
    df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])
    df['clean_message'] = df['message'].apply(clean_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = load_and_train()

st.title('SMS Spam Classifier')

user_input = st.text_area('Enter your SMS message:')

if st.button('Check Spam'):
    if user_input:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        st.write(f'Prediction: **{pred.upper()}**')
    else:
        st.write('Please enter a message.')
