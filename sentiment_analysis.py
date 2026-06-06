import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("reviews.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 Records:")
print(data.head())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

data["clean_text"] = data["text"].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def predict_sentiment(review):

    review = preprocess(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    probabilities = model.predict_proba(review_vector)[0]

    print("\nConfidence Scores")

    for label, score in zip(model.classes_, probabilities):
        print(f"{label}: {score:.2f}")

    return prediction

print("\n========== SAMPLE TESTS ==========")

samples = [
    "The food was amazing and delicious",
    "Average quality and normal service",
    "Worst experience ever"
]

for review in samples:
    result = predict_sentiment(review)
    print(f"\nReview: {review}")
    print("Predicted Sentiment:", result)

print("\n========== LIVE PREDICTION ==========")

while True:

    review = input("\nEnter Review (type exit to quit): ")

    if review.lower() == "exit":
        break

    result = predict_sentiment(review)

    print("Sentiment:", result)