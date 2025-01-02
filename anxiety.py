from fastapi import FastAPI, HTTPException
from datetime import datetime
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import torch

###############################################################################
# 1. Concurrency Control
###############################################################################
DEFAULT_CONCURRENCY = 10
SEMAPHORE = asyncio.Semaphore(DEFAULT_CONCURRENCY)

###############################################################################
# 2. FastAPI Initialization
###############################################################################
app = FastAPI(
    title="Optimized Anxiety Detection Service",
    description="A lightweight service mapping 'fear' to 'anxiety' using an optimized DistilBERT model.",
    version="3.0.0",
)

###############################################################################
# 3. Global Variables
###############################################################################
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
ANXIETY_THRESHOLD = 0.5

emotion_model = None
tokenizer = None

###############################################################################
# 4. Resource Loading at Startup
###############################################################################
@app.on_event("startup")
async def load_resources():
    global emotion_model, tokenizer
    print("Loading resources...")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # Optional: Use GPU if available
        if torch.cuda.is_available():
            emotion_model.to("cuda")
            print("Using GPU for inference.")
        else:
            print("Using CPU for inference.")
        print("Resources loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load resources: {str(e)}")

###############################################################################
# 5. Cached Emotion Detection
###############################################################################
@lru_cache(maxsize=100)
def cached_emotion_detection(text: str):
    """
    Tokenizes the given text, performs emotion classification, and returns
    the probability distribution for each label.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = emotion_model(**inputs)
        probs = outputs.logits.softmax(dim=-1).cpu().numpy()[0]  # shape: (6,)
        return probs
    except Exception as e:
        raise RuntimeError(f"Error during inference: {str(e)}")

###############################################################################
# 6. Main Endpoint: Anxiety Detection
###############################################################################
@app.post("/analyze/anxiety")
async def analyze_anxiety(input_data: dict):
    """
    Analyze the input text for anxiety based on the 'fear' label.
    """
    async with SEMAPHORE:
        if "text" not in input_data:
            raise HTTPException(status_code=400, detail="Invalid input: 'text' is required.")

        text = input_data["text"].strip()
        if not text:
            raise HTTPException(status_code=400, detail="Invalid input: text cannot be empty.")

        # Perform inference
        probs = cached_emotion_detection(text)

        # Map 'fear' to anxiety score
        if "fear" in LABELS:
            fear_idx = LABELS.index("fear")
            anxiety_score = float(probs[fear_idx])
        else:
            anxiety_score = 0.0

        is_anxious = (anxiety_score > ANXIETY_THRESHOLD)

        # Construct response
        return {
            "is_anxious": is_anxious,
            "anxiety_score": anxiety_score,
            "emotion_distribution": probs.tolist(),
            "timestamp": datetime.now().isoformat(),
        }
