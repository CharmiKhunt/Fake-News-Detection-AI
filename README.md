# 🛡️ TRUTHLENS AI
### AI-Powered News Classification & Live Fact Verification System

VeriNews AI is an intelligent Fake News Detection system that combines **Machine Learning**, **Natural Language Processing (NLP)**, and **Live News Verification** to analyze news articles and determine whether they are likely to be **Real** or **Fake**.

Unlike traditional fake news classifiers that rely only on historical datasets, VeriNews AI also supports **Live Fact Verification using News APIs**, allowing users to compare submitted news with current news articles from trusted sources.

---

## 🚀 Features

- 📰 Fake News Detection using Machine Learning
- 🧠 NLP-based Text Preprocessing
- 📊 TF-IDF Feature Extraction
- 🤖 Logistic Regression Classification
- 📈 Prediction Confidence Score
- 📋 Article Statistics
- 📄 PDF Prediction Report Generation
- 📜 Prediction History
- 🌐 Live Fact Verification using News APIs
- 📊 Dashboard with Model Performance
- 🎨 User-friendly Streamlit Interface

---

# 📂 Project Structure

```
VeriNewsAI/
│
├── app.py
├── train.py
├── preprocessing.py
├── report_generator.py
├── visualization.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── outputs/
│   ├── class_distribution.png
│   ├── fake_wordcloud.png
│   └── real_wordcloud.png
│
└── notebooks/
```

---

# 🧠 Machine Learning Pipeline

### 1. Data Collection
- Fake.csv
- True.csv
- Live News API

↓

### 2. Data Cleaning
- Remove duplicates
- Handle missing values

↓

### 3. NLP Preprocessing
- Lowercase Conversion
- URL Removal
- HTML Tag Removal
- Punctuation Removal
- Number Removal
- Stopword Removal
- Porter Stemming

↓

### 4. Feature Engineering
- TF-IDF Vectorization
- Vocabulary Size = 5000

↓

### 5. Model Training
- Logistic Regression

↓

### 6. Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

↓

### 7. Model Deployment
- Streamlit Web Application

↓

### 8. Live Fact Verification
- News API Integration
- Compare submitted article with trusted live news sources

---

# 📊 Dataset

### Training Dataset

- Fake.csv
- True.csv

Combined Dataset

- **44,898 News Articles**

After preprocessing

- **44,689 Articles**

Classes

- Fake News
- Real News

---

# 📈 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 98.58% |
| Precision | 98.9% |
| Recall | 98.9% |
| F1 Score | 98.9% |

---

# 🛠 Technologies Used

### Programming

- Python 3

### Machine Learning

- Scikit-learn
- Logistic Regression

### NLP

- NLTK
- TF-IDF Vectorizer

### Web Framework

- Streamlit

### Visualization

- Matplotlib
- WordCloud

### Report Generation

- ReportLab

### APIs

- Live News API (Fact Verification)

---

# 📦 Python Libraries

- streamlit
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- wordcloud
- joblib
- reportlab
- requests

---


# 🖥 Application Pages

## 🏠 Home

- Predict Fake/Real News
- Confidence Score
- Article Statistics
- Download PDF Report

---

## 📊 Model Performance

Displays

- Dataset Overview
- Accuracy
- Confusion Matrix
- Class Distribution
- Word Clouds

---

## ℹ About

Contains

- Project Description
- Technologies Used
- NLP Pipeline
- Dataset Information
- Developer Details

---

# 📄 PDF Report

The generated report contains

- Prediction Result
- Confidence Score
- Original Article
- Cleaned Article
- Article Statistics
- Timestamp

---

# 🌍 Live Fact Verification

The application can also verify news using trusted online news sources through News APIs.

Features include

- Fetching live news articles
- Comparing user input with current news
- Displaying related news headlines
- Supporting manual fact-checking

---

# 🔮 Future Improvements

- BERT-based Fake News Detection
- RoBERTa Transformer Model
- Explainable AI (SHAP/LIME)
- Multilingual News Detection
- AI-generated News Detection
- Image Verification
- URL Credibility Analysis
- Social Media News Detection
- User Authentication
- Cloud Deployment

---

# 👨‍💻 Developer

**Charmi Khunt**

Computer Engineering Student

Machine Learning • NLP • Data Science

---

