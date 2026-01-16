# Resume Fit Scoring Web App

Resume screening system using Machine Learning with explainable predictions. Upload a PDF resume and get a fit probability score along with the keywords that influenced the decision.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)

## Features

### Resume Upload & Parsing
- **PDF Upload**: Upload resumes in PDF format
- **Text Extraction**: Uses pdfplumber for  text extraction
- **Scanned PDF Detection**: Flags likely scanned documents

### Machine Learning Pipeline
- **TF-IDF Vectorization**: Captures useful phrases with unigrams + bigrams
- **Logistic Regression**: Fast classifier
- **Class-Balanced Training**: Handles imbalanced datasets
- **Probabilistic Output**: Returns confidence scores


### Scoring & Decision Logic
- **Fit Probability Score**: Percentage based scoring
- **Adjustable Threshold**: Set your own strictness level
- **Clear Decision Labels**: "Strong Fit" or "Weak Fit" based on meeting the adjustable threshold
- **Keyword Contribution Analysis**: See which terms influenced the score


### Web Interface
- **Clean Upload UI**: Easy file selection
- **Results Dashboard**: All information centralized
- **Text Demo Mode**: Test with pasted text
- **Mobile Responsive**: Works on all devices

## How to Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

This will:
- Load training data from `data/training_data.csv`
- Train a TF-IDF + Logistic Regression pipeline
- Evaluate using 5-fold cross-validation
- Save the model to `models/resume_model.joblib`

### 3. Run the Web App
```bash
python app.py
```


##  Model Details

### Pipeline Architecture
```
Input Text
    ↓
TfidfVectorizer (unigrams + bigrams)
    ↓
LogisticRegression (class_weight="balanced")
    ↓
Probability Score [0.0, 1.0]
```

### Training Data Format
CSV with columns:
- `resume_text`: The resume content
- `label`: 1 = good fit, 0 = not a good fit

### Explainability Formula
```
contribution = TF-IDF_value × model_coefficient
```

Terms with high TF-IDF values AND positive coefficients contribute most to a "good fit" prediction.

---

<center><p style="color:grey"> Built by Arjun Challa with ❤️ using Flask, scikit-learn, and pdfplumber </p></center>

