import joblib

model = joblib.load("model/ticket_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

print("✅ Model Loaded Successfully!")

while True:
    text = input("\nEnter ticket text (or type exit): ")
    if text.lower() == "exit":
        break

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    print("✅ Predicted Category:", pred)
