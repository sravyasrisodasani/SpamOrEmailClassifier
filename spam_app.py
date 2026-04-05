import streamlit as st
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- NLTK SETUP ----------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------- PAGE STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#667eea,#764ba2);
}
.block-container {
    max-width: 700px;
    margin: auto;
    padding-top: 40px;
}
.title {
    text-align: center;
    color: #4c1d95;
    font-size: 36px;
}
textarea {
    caret-color: black;
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px !important;
    border: 2px solid #7c3aed !important;
    padding: 10px !important;
}
div.stButton > button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #6d28d9;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    return tfidf, model

tfidf, model = load_model()

# ---------- TEXT PROCESSING ----------
def transform_text(text):
    text = text.lower()
    words = text.split()
    filtered = []
    for word in words:
        word = re.sub(r'[^a-zA-Z0-9]', '', word)
        if word and word not in stop_words:
            filtered.append(ps.stem(word))
    return " ".join(filtered)

# ---------- UI ----------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>📩 Email / SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.write("### Enter your message")
input_sms = st.text_area("Message", height=150)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        st.subheader("Prediction Result")

        # Always preprocess text for display
        transformed_sms = transform_text(input_sms)

        # 🔥 Rule-based spam detection
        if any(word in input_sms.lower() for word in ["earn", "work from home", "lakh", "free","exclusive", "per month", "income"]):
            st.error("🚨 This message is SPAM (rule-based)")
        else:
            # ML prediction
            vector_input = tfidf.transform([transformed_sms]).toarray()
            result = model.predict(vector_input)[0]
            prob = model.predict_proba(vector_input)[0][result]

            if result == 1:
                st.error(f"🚨 This message is SPAM\n\nConfidence: {prob*100:.2f}%")
            else:
                st.success(f"✅ This message is NOT SPAM\n\nConfidence: {prob*100:.2f}%")


st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("Built with ❤️ using Python, NLP and Streamlit")