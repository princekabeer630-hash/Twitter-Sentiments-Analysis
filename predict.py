"""
Twitter Sentiment Analysis — Interactive Predictor
===================================================
Run this script after training the model to analyze
your own tweets interactively from the command line.

Usage:
    python src/predict.py
    python src/predict.py --text "I love this product!"
    python src/predict.py --file data/my_tweets.csv
"""

import argparse
import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from sentiment_model import TextPreprocessor, SentimentClassifier, generate_sample_data
from sklearn.model_selection import train_test_split

TEXT_COLUMN_CANDIDATES = ("text", "tweet", "content", "full_text")


def resolve_text_column(columns):
    """Return the best text column for tweet CSV exports."""
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_or_train_model():
    """Load saved model or train a fresh one if not found."""
    model_path = "models/sentiment_model.pkl"
    if os.path.exists(model_path):
        print("📦 Loading saved model...")
        return SentimentClassifier.load(model_path)
    else:
        print("🔧 No saved model found — training a new one...")
        df = generate_sample_data(2000)
        preprocessor = TextPreprocessor()
        df["clean_text"] = preprocessor.transform(df["text"].tolist())
        X_train, _, y_train, _ = train_test_split(
            df["clean_text"], df["label"],
            test_size=0.2, random_state=42, stratify=df["label"]
        )
        clf = SentimentClassifier()
        clf.train(X_train, y_train)
        clf.save(model_path)
        return clf


def print_result(result: dict):
    """Pretty-print a single prediction result."""
    emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}
    colors = {"positive": "\033[92m", "negative": "\033[91m", "neutral": "\033[94m"}
    reset = "\033[0m"
    sentiment = result["sentiment"]
    e = emoji.get(sentiment, "")
    c = colors.get(sentiment, "")
    bar_len = int(result["confidence"] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)

    print(f"\n  Tweet     : {result['text']}")
    print(f"  Sentiment : {c}{sentiment.upper()}{reset} {e}")
    print(f"  Confidence: [{bar}] {result['confidence']:.1%}")
    print(f"  Scores    : 😊 {result['scores']['positive']:.2f}  "
          f"😠 {result['scores']['negative']:.2f}  "
          f"😐 {result['scores']['neutral']:.2f}")
    print("  " + "─" * 55)


def interactive_mode(model):
    """Run an interactive REPL for typing tweets."""
    print("\n" + "=" * 60)
    print("  🐦 Twitter Sentiment Analyzer — Interactive Mode")
    print("  Type a tweet and press Enter.  Type 'quit' to exit.")
    print("=" * 60)
    while True:
        try:
            text = input("\n  Enter tweet > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        if not text:
            continue
        results = model.predict([text])
        print_result(results[0])


def file_mode(model, file_path: str):
    """Analyze all tweets in a CSV file with a common tweet text column."""
    import pandas as pd
    print(f"\n📂 Reading tweets from: {file_path}")
    df = pd.read_csv(file_path)
    text_column = resolve_text_column(df.columns)
    if text_column is None:
        candidates = ", ".join(TEXT_COLUMN_CANDIDATES)
        print(f"❌ CSV must have one of these text columns: {candidates}.")
        return
    results = model.predict(df[text_column].tolist())
    df["predicted_sentiment"] = [r["sentiment"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    out = file_path.replace(".csv", "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"✅ Predictions saved → {out}")
    print(df[[text_column, "predicted_sentiment", "confidence"]].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Predictor")
    parser.add_argument("--text", type=str, help="Single tweet to analyze")
    parser.add_argument("--file", type=str, help="CSV file with a 'text' column")
    args = parser.parse_args()

    model = load_or_train_model()

    if args.text:
        results = model.predict([args.text])
        print_result(results[0])
    elif args.file:
        file_mode(model, args.file)
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
