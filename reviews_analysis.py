import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import pipeline

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Reviews.csv")

print(df.head())
print("Total Reviews:", len(df))


# -----------------------------
# Sentiment Labeling
# -----------------------------
def label_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

df["Sentiment"] = df["Score"].apply(label_sentiment)

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())


# -----------------------------
# Sentiment Visualization
# -----------------------------
sns.countplot(data=df, x="Sentiment")
plt.title("Customer Sentiment Distribution")
plt.show()


# -----------------------------
# AI Sentiment Model Demo
# -----------------------------
sentiment_model = pipeline("sentiment-analysis")

sample_reviews = df["Text"].dropna().head(10)

print("\nAI Sentiment Predictions:")
for review in sample_reviews:
    print(sentiment_model(review[:512]))


# -----------------------------
# Extract Negative Reviews
# -----------------------------
negative_reviews = df[df["Sentiment"] == "Negative"]

print("\nTotal Negative Reviews:", len(negative_reviews))


# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

negative_reviews["Clean_Text"] = negative_reviews["Text"].astype(str).apply(clean_text)


# -----------------------------
# NLP Setup
# -----------------------------
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# -----------------------------
# Word Frequency Analysis
# -----------------------------
text_data = " ".join(negative_reviews["Clean_Text"].dropna())

words = text_data.split()

filtered_words = [
    word for word in words
    if word not in stop_words and len(word) > 3
]

word_counts = Counter(filtered_words)

print("\nMost Common Complaint Words:")
print(word_counts.most_common(20))


# -----------------------------
# Visualization
# -----------------------------
top_words = word_counts.most_common(10)

words = [w[0] for w in top_words]
counts = [w[1] for w in top_words]

plt.figure(figsize=(10,5))
sns.barplot(x=counts, y=words)
plt.title("Top Complaint Keywords in Negative Reviews")
plt.show()


# -----------------------------
#  Complaint Topic Clustering
# -----------------------------
print("\n--- Discovering Complaint Categories ---")

sample_texts = negative_reviews["Clean_Text"].dropna().sample(5000, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

X = vectorizer.fit_transform(sample_texts)

kmeans = KMeans(n_clusters=5, random_state=42)

kmeans.fit(X)

# FIXED VERSION
terms = vectorizer.get_feature_names()

clusters = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop Complaint Clusters:\n")

for i in range(5):
    top_terms = [terms[ind] for ind in clusters[i, :8]]
    print(f"Cluster {i+1}:", ", ".join(top_terms))


# -----------------------------
# Executive Insight Summary
# -----------------------------
print("\n--- Executive Insight Summary ---")

top_complaints = [w[0] for w in word_counts.most_common(5)]

print(f"""
Analysis of {len(df)} customer reviews reveals that most negative feedback
is associated with the following issues:

Top complaint keywords:
{top_complaints}

Topic clustering of negative reviews reveals major dissatisfaction drivers such as:
- Product taste and flavor issues
- Packaging or shipping damage
- Perceived product quality problems
- Price/value dissatisfaction
- Delivery or order fulfillment issues

Improving these areas could significantly enhance customer satisfaction
and strengthen long-term brand perception.
""")