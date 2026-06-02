import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("reviews.csv")

# Remove empty rows
df = df.dropna()

# Convert columns to string
df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess(text):
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# Clean text
df["clean_text"] = df["text"].apply(preprocess)

# Convert text to numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]
print(df[["text","sentiment"]])
print("\n Unique sentiments:",df["sentiment"].unique())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, predictions))

print("\nModel trained successfully!")

# User input loop
while True:
    review = input("\nEnter review (type 'exit' to quit): ")

    if review.lower() == "exit":
        print("Program ended.")
        break

    clean_review = preprocess(review)

    vector = vectorizer.transform([clean_review])

    result = model.predict(vector)

    print("Sentiment:", result[0])

