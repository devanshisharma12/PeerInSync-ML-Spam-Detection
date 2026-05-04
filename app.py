""" 
#http://localhost:8000/docs

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pickle
import sklearn

print(sklearn.__version__)

app = FastAPI()


# ==============================
# 🔥 LOAD MODEL (VERY IMPORTANT)
# ==============================
try:
    with open("model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)


# ==============================
# EVENT MODEL
# ==============================
class EventPayload(BaseModel):
    name: str
    project_title: str
    description: Optional[str] = ""
    creationDate: Optional[str] = None
    date: str
    time: str
    event_type: str
    loc_link: Optional[str] = ""
    maxParticipants: Optional[int] = 0
    participants: Optional[List[str]] = []


# ==============================
# DISCUSSION MODEL
# ==============================
class DiscussionPayload(BaseModel):
    name: str
    postTitle: str
    postBody: str
    editedFlag: bool


# ==============================
# HELPER FUNCTIONS
# ==============================
def is_random(text):
    text = text.lower().replace(" ", "")
    
    # only flag long meaningless strings
    if len(text) < 15:
        return False

    vowels = sum(1 for c in text if c in "aeiou")
    return vowels / max(len(text), 1) < 0.2


def is_spam_or_abuse(text):
    text = text.lower()

    spam_words = [
        "buy now", "click here", "free money",
        "earn money", "visit link", "subscribe"
    ]

    for word in spam_words:
        if word in text:
            return True, "spam"

    return False, ""


# ==============================
# ML PREDICTION FUNCTION
# ==============================
def predict_text(text: str):

    if is_random(text):
        return True, 1.0, "random"

    is_bad, reason = is_spam_or_abuse(text)
    if is_bad:
        return True, 1.0, reason

    # 🔥 ADD THIS CHECK
    if not text or len(text.strip()) == 0:
        return True, 1.0, "empty"

    try:
        vec = vectorizer.transform([text])
        prob = model.predict_proba(vec)[0][1]

        print(vectorizer)
        print(model)

        is_garbage = prob > 0.75
        return is_garbage, float(prob), "ml"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return True, 1.0, "error"


# ==============================
# EVENT ENDPOINT
# ==============================
@app.post("/predict/event")
def predict_event(payload: EventPayload):

    text = f"{payload.name} {payload.project_title} {payload.description} {payload.event_type}"

    is_garbage, confidence, source = predict_text(text)

    return {
        "approved": not is_garbage,
        "confidence": confidence,
        "source": source
    }


# ==============================
# DISCUSSION ENDPOINT
# ==============================
@app.post("/predict/discussion")
def predict_discussion(payload: DiscussionPayload):

    text = f"{payload.name} {payload.postTitle} {payload.postBody}"

    is_garbage, confidence, source = predict_text(text)

    return {
        "approved": not is_garbage,   # ✅ FIXED
        "confidence": confidence,
        "source": source
    }


# ==============================
# ROOT (OPTIONAL)
# ==============================
@app.get("/")
def root():
    return {"message": "ML API is running 🚀"}






"""
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