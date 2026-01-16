import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("data/ticket.csv")

X = df["text"].astype(str)
y = df["category"].astype(str)

# Convert text -> numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, "model/ticket_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

print("✅ Training complete!")
print("✅ Saved model: model/ticket_model.joblib")
print("✅ Saved vectorizer: model/tfidf_vectorizer.joblib")
