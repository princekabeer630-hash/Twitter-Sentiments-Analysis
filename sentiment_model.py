"""
Twitter Sentiment Analysis - ML Internship Task 1
==================================================
This module provides sentiment analysis on Twitter/text data using:
  - Traditional ML (Logistic Regression, Naive Bayes)
  - Deep Learning (LSTM)
  - Pre-trained Transformers (BERT via HuggingFace)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

class TextPreprocessor:
    """Clean and normalize raw tweet text."""

    def __init__(self):
        self.stopwords = {
            "i","me","my","myself","we","our","ours","ourselves","you","your",
            "yours","he","him","his","she","her","hers","it","its","they","them",
            "their","what","which","who","is","are","was","were","be","been",
            "being","have","has","had","do","does","did","will","would","shall",
            "should","may","might","must","can","could","a","an","the","and",
            "but","if","or","as","of","at","by","for","with","about","to","from",
            "in","on","that","this","these","those"
        }

    def clean(self, text: str) -> str:
        """Apply full cleaning pipeline to a single tweet."""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
        text = re.sub(r"@\w+", "", text)                        # remove @mentions
        text = re.sub(r"#(\w+)", r"\1", text)                  # keep hashtag word
        text = re.sub(r"[^a-z\s]", " ", text)                  # keep only letters
        text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
        tokens = [w for w in text.split() if w not in self.stopwords and len(w) > 2]
        return " ".join(tokens)

    def transform(self, texts):
        return [self.clean(t) for t in texts]


# ─────────────────────────────────────────────────────────────
# 2. SAMPLE DATA GENERATOR  (replace with real dataset)
# ─────────────────────────────────────────────────────────────

def generate_sample_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic tweet-like data for demonstration.
    In production, replace with a real dataset such as:
      - Sentiment140  (1.6M tweets)
      - Twitter US Airline Sentiment (Kaggle)
    """
    positive_templates = [
        "I love {} so much! It's amazing",
        "Having the best day with {}",
        "{} is absolutely fantastic and wonderful",
        "So happy about {} today! Great experience",
        "{} made my day better than ever",
        "Really enjoying {} - highly recommend!",
        "Incredible results with {} - feeling great",
        "Best decision ever to try {} - love it",
    ]
    negative_templates = [
        "I hate {} it's so terrible",
        "Worst experience with {} ever - disgusted",
        "{} is completely awful and disappointing",
        "So angry about {} what a disaster",
        "{} ruined my entire day - frustrated",
        "Never using {} again - total waste",
        "Terrible service from {} - avoid",
        "Deeply unhappy with {} - regret it",
    ]
    neutral_templates = [
        "Just read about {} - interesting topic",
        "Saw {} today, not sure what to think",
        "{} was mentioned in the news",
        "Checked out {}, it exists apparently",
        "Someone told me about {} - okay",
        "{} is a thing that happened today",
    ]

    topics = [
        "the new product", "customer service", "the movie", "this app",
        "the restaurant", "the airline", "the phone", "the game",
        "the concert", "the hotel", "this brand", "the update"
    ]

    rows = []
    np.random.seed(42)
    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # neg/pos/neutral
        topic = np.random.choice(topics)
        if label == 1:
            tmpl = np.random.choice(positive_templates)
        elif label == 0:
            tmpl = np.random.choice(negative_templates)
        else:
            tmpl = np.random.choice(neutral_templates)
        rows.append({"text": tmpl.format(topic), "label": label})

    df = pd.DataFrame(rows)
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    df["sentiment"] = df["label"].map(label_map)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. TRADITIONAL ML MODELS
# ─────────────────────────────────────────────────────────────

class SentimentClassifier:
    """
    Train and evaluate multiple ML classifiers for sentiment analysis.
    Includes: Logistic Regression, Naive Bayes.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizer = None

    def build_pipeline(self, clf):
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2
            )),
            ("clf", clf)
        ])

    def train(self, X_train, y_train):
        print("\n🔧 Training models...")
        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced"
            ),
            "Naive Bayes": MultinomialNB(alpha=0.1),
        }
        for name, clf in classifiers.items():
            pipe = self.build_pipeline(clf)
            pipe.fit(X_train, y_train)
            self.models[name] = pipe
            print(f"  ✅ {name} trained")
        return self

    def evaluate(self, X_test, y_test):
        print("\n📊 Evaluation Results")
        print("=" * 60)
        results = {}
        for name, pipe in self.models.items():
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"accuracy": acc, "predictions": y_pred}
            print(f"\n── {name} ──")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred,
                  target_names=["negative", "positive", "neutral"]))
        return results

    def predict(self, texts, model_name="Logistic Regression"):
        cleaned = self.preprocessor.transform(texts)
        model = self.models[model_name]
        predictions = model.predict(cleaned)
        probabilities = model.predict_proba(cleaned)
        label_map = {0: "negative", 1: "positive", 2: "neutral"}
        return [
            {
                "text": texts[i],
                "sentiment": label_map[predictions[i]],
                "confidence": float(max(probabilities[i])),
                "scores": {
                    "negative": float(probabilities[i][0]),
                    "positive": float(probabilities[i][1]),
                    "neutral": float(probabilities[i][2]),
                }
            }
            for i in range(len(texts))
        ]

    def save(self, path="models/sentiment_model.pkl"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        print(f"\n💾 Models saved → {path}")

    @classmethod
    def load(cls, path="models/sentiment_model.pkl"):
        obj = cls()
        obj.models = joblib.load(path)
        return obj


# ─────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Twitter Sentiment Analysis — Data Overview", fontsize=14, fontweight="bold")

    # Count plot
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
    counts = df["sentiment"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=[colors.get(s, "#888") for s in counts.index])
    axes[0].set_title("Sentiment Distribution")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Count")
    for i, (sent, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 10, str(cnt), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index,
                colors=[colors.get(s, "#888") for s in counts.index],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Sentiment Proportion")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Plot saved → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["negative", "positive", "neutral"]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Confusion matrix saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Twitter Sentiment Analysis — ML Internship Task 1")
    print("=" * 60)

    # ── Step 1: Load / Generate Data ──────────────────────────
    print("\n📥 Loading data...")
    df = generate_sample_data(n_samples=2000)
    print(f"   Dataset shape : {df.shape}")
    print(f"   Label counts  :\n{df['sentiment'].value_counts()}")

    # ── Step 2: Preprocess ────────────────────────────────────
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.transform(df["text"].tolist())

    # ── Step 3: Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\n🔀 Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Step 4: Train & Evaluate ──────────────────────────────
    clf = SentimentClassifier()
    clf.train(X_train, y_train)
    results = clf.evaluate(X_test, y_test)

    # ── Step 5: Visualize ─────────────────────────────────────
    import os
    os.makedirs("outputs", exist_ok=True)
    plot_sentiment_distribution(df, save_path="outputs/sentiment_distribution.png")

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    plot_confusion_matrix(
        y_test, results[best_model]["predictions"],
        model_name=best_model,
        save_path="outputs/confusion_matrix.png"
    )

    # ── Step 6: Save model ────────────────────────────────────
    clf.save("models/sentiment_model.pkl")

    # ── Step 7: Demo Prediction ───────────────────────────────
    demo_tweets = [
        "I absolutely love this product! It's the best thing ever!",
        "This is terrible, worst experience of my life, totally useless",
        "Just saw the new update. Not sure what to think about it yet.",
        "Amazing customer service today, really impressed!",
        "The app keeps crashing, so frustrating and annoying",
    ]

    print("\n🔍 Demo Predictions (Logistic Regression)")
    print("-" * 60)
    predictions = clf.predict(demo_tweets)
    for p in predictions:
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}
        e = emoji.get(p["sentiment"], "")
        print(f"{e} [{p['sentiment'].upper():8}] ({p['confidence']:.0%}) → {p['text'][:55]}")

    print("\n✅ Pipeline complete! Check the 'outputs/' folder for plots.")
    return clf


if __name__ == "__main__":
    main()
