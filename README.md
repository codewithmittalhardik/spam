# 📧 SMS/Email Spam Classifier

A machine learning-based web application that classifies SMS or Email messages as **Spam** or **Not Spam (Ham)**. The app uses a pre-trained model to analyze text in real-time and provide instant predictions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 🚀 Features
- **Real-time Prediction**: Instantly classifies messages as you type or paste them.
- **Text Preprocessing**: Automatically handles tokenization, stopword removal, and stemming.
- **Simple UI**: Built with Streamlit for a clean and responsive user interface.
- **Model Transparency**: Uses a pre-trained Naive Bayes/Machine Learning model (`model.pkl`).

## 🛠️ Tech Stack
- **Language**: Python
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: Scikit-learn
- **NLP**: NLTK (Natural Language Toolkit)
- **Data Handling**: Pandas, NumPy

Run these commands in your terminal to set up and start the project:

```bash
# 1. Clone the repo
git clone [https://github.com/codewithmittalhardik/spam.git](https://github.com/codewithmittalhardik/spam.git)
cd spam
```

# 2. Create & Activate Virtual Environment
```bash
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

# 3. Install Dependencies
```bash
pip install -r requirements.txt
```

# 4. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure
```text
spam/
├── app.py              # Main Streamlit application file
├── model.pkl           # Pre-trained machine learning model
├── vectorizer.pkl      # Pre-trained TF-IDF/CountVectorizer
├── requirements.txt    # List of python dependencies
└── README.md           # Project documentation
```
