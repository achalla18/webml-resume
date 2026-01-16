import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

DATA_PATH = "data/training_data.csv"
MODEL_PATH = "models/resume_model.joblib"
RANDOM_STATE = 42


def load_data(path: str) -> tuple[pd.Series, pd.Series]:
    """Load and validate training data."""
    print(f"Loading data from {path}...")
    data = pd.read_csv(path)
    
    required_cols = ["resume_text", "label"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean
    data = data.dropna(subset=required_cols)
    data["resume_text"] = data["resume_text"].str.strip()
    data = data[data["resume_text"].str.len() > 0]
    
    print(f"Loaded {len(data)} samples")
    print(f"Class distribution:\n{data['label'].value_counts()}")
    
    return data["resume_text"], data["label"]


def create_pipeline() -> Pipeline:

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",       # Remove common English 
            ngram_range=(1, 2),         # Unigrams + Bigrams
            max_df=0.95,                
            min_df=2,                   
            sublinear_tf=True,          
            lowercase=True,             
            max_features=5000           
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",   
            max_iter=1000,              
            random_state=RANDOM_STATE,
            C=1.0                       
        ))
    ])
    
    return pipeline


def evaluate_model(pipeline: Pipeline, X: pd.Series, y: pd.Series) -> dict:
    """Evaluate model using cross-validation and holdout set."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"\n5-Fold Cross-Validation Accuracy:")
    print(f"  Mean: {cv_scores.mean():.3f}")
    print(f"  Std:  {cv_scores.std():.3f}")
    print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\nHoldout Set Performance (20% test):")
    print(classification_report(y_test, y_pred, target_names=["Not Fit", "Good Fit"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    
    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }


def get_feature_importance(pipeline: Pipeline, top_n: int = 20) -> dict:
    """Extract top positive and negative feature weights."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    sorted_idx = np.argsort(coefficients)
    
    # Top positive (good fit indicators)
    top_positive = [
        (feature_names[i], coefficients[i]) 
        for i in sorted_idx[-top_n:][::-1]
    ]
    
    # Top negative (not fit indicators)
    top_negative = [
        (feature_names[i], coefficients[i]) 
        for i in sorted_idx[:top_n]
    ]
    
    print("\n" + "="*50)
    print("TOP FEATURE WEIGHTS")
    print("="*50)
    
    print("\nTop 'Good Fit' Indicators:")
    for term, weight in top_positive[:10]:
        print(f"  {term:30} {weight:+.3f}")
    
    print("\nTop 'Not Fit' Indicators:")
    for term, weight in top_negative[:10]:
        print(f"  {term:30} {weight:+.3f}")
    
    return {
        "positive": top_positive,
        "negative": top_negative
    }


def save_model(pipeline: Pipeline, path: str):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\nModel saved to: {path}")
    
    # Verify save
    file_size = os.path.getsize(path) / 1024
    print(f"File size: {file_size:.1f} KB")


def main():
    """Main training workflow."""
    print("="*50)
    print("RESUME FIT MODEL TRAINING")
    print("="*50)
    
    X, y = load_data(DATA_PATH)
    
    pipeline = create_pipeline()
    
    metrics = evaluate_model(pipeline, X, y)
    
    print("\n" + "="*50)
    print("Training Final Model on Full Dataset")
    print("="*50)
    pipeline.fit(X, y)
    
    get_feature_importance(pipeline)
    
    save_model(pipeline, MODEL_PATH)
    
    print("\n" + "="*50)
    print("Test Predictions")
    print("="*50)
    
    test_resumes = [
        "Python machine learning data science pandas sklearn tensorflow neural networks",
        "Customer service retail sales cashier experience Excel spreadsheets",
        "Senior data scientist with PhD in statistics, expert in deep learning and NLP"
    ]
    
    for resume in test_resumes:
        prob = pipeline.predict_proba([resume])[0][1]
        label = "Strong Fit" if prob >= 0.7 else "Weak Fit"
        print(f"\n'{resume[:60]}...'")
        print(f"  Probability: {prob:.1%} -> {label}")
    
    print("\n" + "="*50)
    print("Training Cycle Complete")
    print("="*50)


if __name__ == "__main__":
    main()
