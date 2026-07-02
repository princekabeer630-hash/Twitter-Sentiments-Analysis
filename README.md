# Twitter-Sentiments-Analysis

🐦 Twitter Sentiment Analysis using Machine Learning — Classifies tweets as Positive, Negative &amp; Neutral using TF-IDF + Logistic Regression &amp; Naive Bayes | ML Internship Task 1 by RhombixTechnologies

## CSV Prediction

After training, run predictions for a CSV file:

```bash
python predict.py --file data/my_tweets.csv
```

The predictor automatically uses the first matching text column from:

- `text`
- `tweet`
- `content`
- `full_text`

That supports common exports from tweet collection tools, including TweetClaw CSV
exports, without renaming columns first.
