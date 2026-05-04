from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

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

def predict_text(text: str):
    # replace this with your trained model logic
    # return (is_garbage, confidence, source)
    return False, 0.12, "ml"

@app.post("/predict")
def predict_event(payload: EventPayload):
    text = f"{payload.name} {payload.project_title} {payload.description} {payload.event_type}"

    is_garbage, confidence, source = predict_text(text)

    return {
        "approved": is_garbage,
        "confidence": confidence,
        "source": source
    }