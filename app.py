import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- SAFETY CHECK: Download NLTK data automatically ---
# This prevents "LookupError" if run on a new machine
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

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

# Load the models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the Message')

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    # 2. Vectorize
    # New line (Fixes the error)
    vector_input = tfidf.transform([transform_sms]).toarray()
    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display Result (Using header for better visibility)
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')

st.write("Here are some examples you can try:")
st.write("1. Congratulations, you won 1000 calls on this number to get your prize.")
st.write("2. Hey man, are we still on for the movie tonight? I'm running a bit late but should be there by 7.")
st.write("3. Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's.")
st.write("4. Your Uber ride is arriving in 2 minutes. Look for a white Toyota Camry with license plate ABC-1234.")

# --- SOCIAL LINKS FOOTER ---
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 20px;
        z-index: 1000;
    }
    .footer a {
        margin: 0 15px;
        color: #333;
        text-decoration: none;
    }
    .footer .linkedin:hover {
        color: #0072b1;
        transform: scale(1.2);
    }.footer .github:hover {
        color: #f1f1f1;
        transform: scale(1.2);
    }.footer .mail:hover {
        color: red;
        transform: scale(1.2);
    }
    </style>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <div class="footer">
        <p>Connect with me:</p>
        <a href="https://www.linkedin.com/in/mittalhardik" target="_blank" class="linkedin"><i class="fab fa-linkedin"></i></a>
        <a href="https://github.com/codewithmittalhardik" target="_blank" class="github"><i class="fab fa-github"></i></a>
        <a href="mailto:mittalhardik2007@gmail.com" class="mail"><i class="fas fa-envelope"></i></a>
    </div>
    """, unsafe_allow_html=True)