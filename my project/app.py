import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the pre-trained TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Load the pre-trained MultinomialNB model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Debug prints
    print("Transformed Message:", transformed_sms)

    # Vectorize the input message
    vector_input = tfidf.transform([transformed_sms])

    # Debug prints
    print("Vectorized Input:", vector_input)

    # Make predictions
    result = model.predict(vector_input)

    # Debug prints
    print("Prediction Result:", result)

    # Display the result
    if result[0] == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
