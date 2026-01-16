import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model + vectorizer
model = joblib.load("model/ticket_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

app = FastAPI(title="SupportDesk AI - Ticket Classifier")

# Input format for API
class TicketRequest(BaseModel):
    text: str


def predict_priority(text: str) -> str:
    t = text.lower()

    high_keywords = ["urgent", "refund", "immediately", "asap", "charged twice"]
    medium_keywords = ["error", "issue", "crash", "not working", "failed"]
    low_keywords = ["how", "pricing", "information", "details"]

    if any(k in t for k in high_keywords):
        return "High"
    if any(k in t for k in medium_keywords):
        return "Medium"
    if any(k in t for k in low_keywords):
        return "Low"
    return "Medium"


def auto_reply(category: str) -> str:
    replies = {
        "Billing": "Sorry for the inconvenience. Please share your payment/transaction details and we will assist you.",
        "Refund": "We’re sorry to hear that. Please share your order/payment details so we can process your refund request.",
        "Technical": "Thanks for reporting this issue. Please share a screenshot/error message and your device details.",
        "Account": "We can help you recover your account. Please confirm your registered email/phone number.",
        "Delivery": "Sorry for the delivery issue. Please share your order ID and we will check the delivery status.",
        "General": "Thanks for reaching out! Please share more details and we’ll help you.",
    }
    return replies.get(category, "Thanks for reaching out! Our support team will contact you soon.")


@app.get("/")
def home():
    return {"message": "SupportDesk AI is running ✅. Open /docs to test the API."}


@app.post("/predict")
def predict_ticket(req: TicketRequest):
    text = req.text

    X = vectorizer.transform([text])
    category = model.predict(X)[0]

    confidence = float(model.predict_proba(X).max())

    priority = predict_priority(text)
    reply = auto_reply(category)

    return {
        "category": category,
        "priority": priority,
        "confidence": round(confidence, 3),
        "suggested_reply": reply
    }
