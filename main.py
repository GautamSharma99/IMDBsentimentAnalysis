#step1: Import library and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model

# load the imdb dataset word index

word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

#load the pre-trained model with reLU activation
model=load_model('simple_rnn.h5')

#step2: helper functions
#decode function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

#preprocessing function
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


# step3: prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

#streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or neagtive')

#user input
user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Postive' if prediction[0][0] > 0.5 else 'Negative'


    #display
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

