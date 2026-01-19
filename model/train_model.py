import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("data/tickets.csv")  # ✅ your file name is ticket.csv

X = df["text"].astype(str)
y = df["category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("✅ Accuracy:", round(accuracy_score(y_test, preds), 4))
print("\n✅ Classification Report:\n")
print(classification_report(y_test, preds))

joblib.dump(model, "model/ticket_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

print("\n✅ Saved model: model/ticket_model.joblib")
print("✅ Saved vectorizer: model/tfidf_vectorizer.joblib")
