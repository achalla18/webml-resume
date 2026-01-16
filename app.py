import os
import uuid
from typing import Optional
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
import pdfplumber
import joblib

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MODEL_PATH = "models/resume_model.joblib"
DEFAULT_THRESHOLD = 0.70
MIN_TEXT_LENGTH = 50  # Minimum chars 

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model at startup (production pattern)
model = None


def load_model():
    """Load the trained model. Called once at startup."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}. Run train_model.py first.")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath: str) -> tuple[str, bool]:
    """
    Extract text from PDF file using pdfplumber.
    
    Returns:
        tuple: (extracted_text, is_likely_scanned)
    """
    text_parts = []
    
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return "", True
    
    full_text = "\n".join(text_parts)
    
    full_text = " ".join(full_text.split())
    
    # Check if likely scanned 
    is_likely_scanned = len(full_text.strip()) < MIN_TEXT_LENGTH
    
    return full_text, is_likely_scanned


def get_keyword_contributions(resume_text: str, top_n: int = 10) -> list[dict]:
    """
    Calculate per-term contribution to the prediction.
    
    Contribution ≈ TF-IDF value × model coefficient
    
    This provides explainability for why a resume scored the way it did.
    """
    if model is None:
        return []
    
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    
    tfidf_vector = vectorizer.transform([resume_text])
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    contributions = []
    tfidf_array = tfidf_vector.toarray()[0]
    
    for idx, (tfidf_val, coef) in enumerate(zip(tfidf_array, coefficients)):
        if tfidf_val > 0:  
            contribution = tfidf_val * coef
            contributions.append({
                "term": feature_names[idx],
                "tfidf": tfidf_val,
                "coefficient": coef,
                "contribution": contribution
            })
    
    # Sort by contribution (positive = good fit)
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    
    top_positive = [c for c in contributions if c["contribution"] > 0][:top_n]
    
    return top_positive


def score_resume(resume_text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Score a resume and return detailed results.
    
    Returns:
        dict with probability, decision, contributing terms, etc.
    """
    if model is None:
        return {
            "error": "Model not loaded. Please run train_model.py first.",
            "probability": 0,
            "decision": "ERROR",
            "threshold": threshold,
            "contributions": []
        }
    
    # Get probability
    probability = model.predict_proba([resume_text])[0][1]
    
    if probability >= threshold:
        decision = "STRONG FIT"
        decision_class = "strong"
    else:
        decision = "WEAK FIT"
        decision_class = "weak"
    
    contributions = get_keyword_contributions(resume_text)
    
    return {
        "probability": probability,
        "probability_percent": f"{probability:.1%}",
        "decision": decision,
        "decision_class": decision_class,
        "threshold": threshold,
        "threshold_percent": f"{threshold:.0%}",
        "contributions": contributions,
        "error": None
    }


@app.route("/", methods=["GET"])
def index():
    """Home page with upload form."""
    return render_template("index.html", default_threshold=int(DEFAULT_THRESHOLD * 100))


@app.route("/score", methods=["POST"])
def score():
    """Handle resume upload and scoring."""
    # Get threshold from form
    try:
        threshold = int(request.form.get("threshold", DEFAULT_THRESHOLD * 100)) / 100
        threshold = max(0.1, min(0.99, threshold))  # Clamp 
    except (ValueError, TypeError):
        threshold = DEFAULT_THRESHOLD
    
    if "resume" not in request.files:
        flash("No file uploaded. Please select a PDF file.", "error")
        return redirect(url_for("index"))
    
    file = request.files["resume"]
    
    if file.filename == "":
        flash("No file selected. Please choose a PDF file.", "error")
        return redirect(url_for("index"))
    
    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PDF file.", "error")
        return redirect(url_for("index"))
    
    # Save file 
    unique_filename = f"{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    
    try:
        file.save(filepath)
        
        resume_text, is_scanned = extract_text_from_pdf(filepath)
        
        if is_scanned or len(resume_text.strip()) < MIN_TEXT_LENGTH:
            flash(
                "Could not extract enough text from PDF. "
                "This may be a scanned document. Please upload a text-based PDF.",
                "error"
            )
            return redirect(url_for("index"))
        
        result = score_resume(resume_text, threshold)
        
        if result["error"]:
            flash(result["error"], "error")
            return redirect(url_for("index"))
        
        return render_template(
            "result.html",
            result=result,
            resume_text=resume_text,
            original_filename=file.filename
        )
        
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for("index"))
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/demo", methods=["GET", "POST"])
def demo():
    """Demo page for testing with text input (no PDF needed)."""
    if request.method == "GET":
        return render_template("demo.html", default_threshold=int(DEFAULT_THRESHOLD * 100))
    
    # form submission
    resume_text = request.form.get("resume_text", "").strip()
    
    if not resume_text:
        flash("Please enter some resume text.", "error")
        return redirect(url_for("demo"))
    
    try:
        threshold = int(request.form.get("threshold", DEFAULT_THRESHOLD * 100)) / 100
        threshold = max(0.1, min(0.99, threshold))
    except (ValueError, TypeError):
        threshold = DEFAULT_THRESHOLD
    
    result = score_resume(resume_text, threshold)
    
    if result["error"]:
        flash(result["error"], "error")
        return redirect(url_for("demo"))
    
    return render_template(
        "result.html",
        result=result,
        resume_text=resume_text,
        original_filename="Text Input"
    )


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash("File too large. Maximum size is 16MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(500)
def server_error(e):
    """Handle server errors."""
    flash("An internal error occurred. Please try again.", "error")
    return redirect(url_for("index"))


load_model()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
