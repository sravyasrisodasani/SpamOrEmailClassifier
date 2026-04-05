# 📩 Spam Message Classifier

## 🚀 Overview

A machine learning based **Spam Classifier** that detects whether a message is **Spam or Not Spam** using NLP techniques and a hybrid approach.

---

## 💥 Key Highlights

* Built an **end-to-end NLP pipeline** from raw text to deployment
* Implemented **text preprocessing** (cleaning, stopwords removal, stemming)
* Preserved **important tokens like numbers (₹, amounts)** for better spam detection
* Compared **Bag of Words vs TF-IDF** → selected **TF-IDF** for better feature representation
* Experimented with multiple models (**Logistic Regression, SVM, Naive Bayes**)
* Achieved **best performance using Multinomial Naive Bayes (MNB)**
* Reached **~96% accuracy and ~1.0 precision** (very low false positives)
* Added **n-grams (1,2)** to capture context like “work from home”, “earn money”
* Solved real-world edge cases using a **Hybrid Model (ML + Rule-Based filtering)**
* Built and deployed an **interactive UI using Streamlit**

---

## 🧠 Model Decisions (Why these choices?)

* **TF-IDF over Bag of Words** → captures importance of words, reduces noise
* **Multinomial Naive Bayes** → best suited for text frequency data
* Other models (Logistic, SVM) tested but **MNB gave highest precision**
* **n-grams** improved detection of phrase-based spam
* Added **rule-based layer** to handle unseen patterns (e.g., job scams)

---

## ⚙️ Hybrid Intelligence

ML alone failed on unseen patterns like:

> “Work from home and earn ₹1 lakh per month”

Solved using rule-based filtering:

* Detects keywords like: *earn, income, lakh, work from home*
* Improves real-world robustness

---

## 📊 Performance

* Accuracy: ~96%
* Precision: ~1.0
* Very low false positives

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas, NumPy
* Streamlit

---

## ▶️ Run Locally

```bash
python train.py
streamlit run spam_app.py
```

---

## 🎯 Outcome

A **production-style spam detection system** combining:

* Machine Learning
* Feature Engineering
* Rule-Based Intelligence

---

## 🧠 Learning

* Importance of **preprocessing consistency**
* Feature selection impacts model performance
* Real-world systems require **hybrid approaches**

---

## ❤️ Built With

Python, NLP, Machine Learning & Streamlit

## 🧠 Technologies Used
- Python
- Streamlit
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Multinomial Naive Bayes

## 📂 Project Structure
spam_classifier
│
├── spam_app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
