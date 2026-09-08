# 📰 Fake News Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify news articles as **Real** or **Fake**.

---

## 📌 Project Overview

With the widespread proliferation of digital media, misinformation poses significant challenges. This project provides a comparative NLP pipeline comparing multiple text preprocessing approaches and classification algorithms on a dataset of **20,800 news articles**.

### Key Highlights:
- **Comprehensive Text Preprocessing**: Evaluates and compares **Porter Stemming** vs. **WordNet Lemmatization** (with POS tagging).
- **TF-IDF Feature Extraction**: Unigram & Bigram representation with sublinear TF scaling.
- **Multiple Classifiers**:
  - Logistic Regression
  - Support Vector Classifier (Linear SVM with Platt calibration)
  - Random Forest Classifier (Ensemble Trees)
- **Extensive Model Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrices
  - Receiver Operating Characteristic (ROC) Curves & AUC Scores
  - Cross-entropy Log-loss (Train vs. Test)
  - Word Clouds & Frequency Analysis
- **Model Serialization & Deployment**: Pre-trained pipelines saved using `joblib` for inference on new articles.

---

## 📊 Performance Comparison

| Model | Preprocessing | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Logistic Regression** | Stemming | 91.95% | 90.79% | 93.37% | 92.06% |
| **Logistic Regression** | Lemmatization | 92.14% | 91.48% | 92.93% | 92.20% |
| **Support Vector Machine (SVM)** | Stemming | 93.94% | 93.30% | 94.67% | 93.98% |
| **Support Vector Machine (SVM)** | Lemmatization | **94.21%** | **94.02%** | **94.42%** | **94.22%** |
| **Random Forest** | Stemming | 91.27% | 92.65% | 89.67% | 91.14% |
| **Random Forest** | Lemmatization | 91.68% | 93.59% | 89.47% | 91.49% |

---

## 📁 Repository Structure

```
Fake News Detection/
├── Fake_news_detection.ipynb  # Complete Jupyter Notebook with code and rendered visualizations
├── sample_news.csv            # Sample dataset preview (100 rows)
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Manmeet2456/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook Fake_news_detection.ipynb
```

---

## 🧪 Interactive Inference Example

```python
import joblib

# Load trained model and vectorizer
model = joblib.load('svm_lem_model.joblib')
vectorizer = joblib.load('vectorizer_lem.joblib')

headline = "Breaking: Major scientific discovery announced today"
processed_features = vectorizer.transform([headline])
prediction = model.predict(processed_features)

print("Result:", "Fake" if prediction[0] == 1 else "Real")
```

---

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3.12+
- **Machine Learning**: `scikit-learn`, `joblib`
- **NLP**: `nltk`, `autocorrect`
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
