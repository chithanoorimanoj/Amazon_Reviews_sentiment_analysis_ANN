import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Amazon Review Sentiment Analysis", page_icon="⭐", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

@st.cache_resource
def load_nltk_resources():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name} in {BASE_DIR}. "
            "Run the updated notebook's training and export cells, then copy model.joblib here."
        )
    artifact = joblib.load(MODEL_PATH)
    # The updated notebook exports a dictionary containing the fitted TF-IDF + Logistic Regression pipeline.
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    # Allow a directly saved fitted sklearn Pipeline as well.
    return {
        "model": artifact,
        "sentiment_map": {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"},
        "metrics": {},
        "best_parameters": {},
    }

def clean_review(text, stop_words, lemmatizer):
    # Must match the updated notebook's clean() function exactly.
    text = re.sub(r"[^a-zA-Z.]", " ", str(text)).lower()
    tokens = word_tokenize(text)
    negation_words = {"no", "not", "nor", "never", "n't"}
    tokens = [word for word in tokens if word not in stop_words or word in negation_words]
    return " ".join(lemmatizer.lemmatize(token) for token in tokens)

st.title("⭐ Amazon Review Sentiment Analysis")
st.write("Enter an Amazon product review to predict its rating (1–5) and sentiment using the trained TF-IDF + Logistic Regression model.")

try:
    artifact = load_model_artifact()
    stop_words, lemmatizer = load_nltk_resources()
except Exception as exc:
    st.error("Could not load the model or text-processing resources.")
    st.code(str(exc))
    st.stop()

st.caption(f"Loaded model file: {MODEL_PATH.name}")
review = st.text_area("Enter your review", placeholder="Example: The product quality is excellent and delivery was fast.", height=150)

if st.button("Analyze Review", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned_review = clean_review(review, stop_words, lemmatizer)
        if not cleaned_review.strip():
            st.warning("The review has no usable words after preprocessing. Please enter more text.")
        else:
            model = artifact["model"]
            try:
                prediction = int(model.predict([cleaned_review])[0])
            except Exception as exc:
                st.error("Prediction failed. Make sure model.joblib was created by the updated notebook and that training completed successfully.")
                st.code(str(exc))
                st.stop()
            sentiment_map = artifact.get("sentiment_map", {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"})
            sentiment = sentiment_map.get(prediction, sentiment_map.get(str(prediction), "Unknown"))
            st.subheader("Prediction")
            c1, c2 = st.columns(2)
            c1.metric("Predicted Rating", f"{prediction} / 5 ⭐")
            c2.metric("Predicted Sentiment", sentiment)
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba([cleaned_review])[0]
                    classes = [str(int(x)) for x in model.classes_]
                    st.write("**Rating probabilities (%)**")
                    st.bar_chart({label: round(float(prob) * 100, 2) for label, prob in zip(classes, probabilities)})
                except Exception:
                    pass
            with st.expander("View preprocessed review"):
                st.write(cleaned_review)

with st.sidebar:
    st.header("About the project")
    st.write("**Feature extraction:** TF-IDF")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Task:** Predict review rating (1–5)")
    metrics = artifact.get("metrics", {})
    if metrics:
        st.divider()
        st.subheader("Saved test metrics")
        for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "Weighted F1 score")]:
            if key in metrics:
                st.metric(label, f"{float(metrics[key]) * 100:.2f}%")
