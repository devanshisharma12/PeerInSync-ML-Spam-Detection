from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pickle

app = FastAPI()

# ==============================
# LOAD MODEL
# ==============================
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

print("✅ Model loaded")

# ==============================
# MODELS
# ==============================

class EventPayload(BaseModel):
    name: str
    project_title: str
    description: Optional[str] = ""
    date: str
    time: str
    event_type: str


class DiscussionPayload(BaseModel):
    name: str
    postTitle: str
    postBody: str


# ==============================
# HELPERS
# ==============================

def is_random(text):
    text = text.lower().replace(" ", "")
    if len(text) < 15:
        return False
    vowels = sum(1 for c in text if c in "aeiou")
    return vowels / max(len(text), 1) < 0.2


def is_spam(text):
    spam_words = [
        "buy now", "click here", "free money",
        "earn money", "visit link", "subscribe"
    ]
    text = text.lower()
    return any(word in text for word in spam_words)


# ==============================
# ML FUNCTION
# ==============================

def predict_text(text: str):

    # Rule checks
    if is_random(text):
        return False, 1.0, "random"

    if is_spam(text):
        return False, 1.0, "spam"

    # ML prediction
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0][1]

    is_garbage = prob > 0.75

    return not is_garbage, float(prob), "ml"


# ==============================
# EVENT API
# ==============================

@app.post("/predict/event")
def predict_event(payload: EventPayload):

    text = f"{payload.name} {payload.project_title} {payload.description} {payload.event_type}"

    approved, confidence, source = predict_text(text)

    return {
        "approved": approved,
        "confidence": confidence,
        "source": source
    }


# ==============================
# DISCUSSION API
# ==============================

@app.post("/predict/discussion")
def predict_discussion(payload: DiscussionPayload):

    text = f"{payload.name} {payload.postTitle} {payload.postBody}"

    approved, confidence, source = predict_text(text)

    return {
        "approved": approved,
        "confidence": confidence,
        "source": source
    }


# ==============================
# ROOT
# ==============================

@app.get("/")
def root():
    return {"message": "ML API running 🚀"}
