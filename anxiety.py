from fastapi import FastAPI, HTTPException
from datetime import datetime
import asyncio
import logging
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    description=(
        "A FastAPI service mapping 'fear' to 'anxiety' using an optimized emotion "
        "model, improved thresholds, and enhanced multilingual support."
    ),
    version="2.1.0",
)

###############################################################################
# 3. Logging Configuration
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

###############################################################################
# 4. Global Variables
###############################################################################
emotion_model = None
emotion_tokenizer = None

LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
ANXIETY_THRESHOLD = 0.3  # Lowered threshold for more sensitive detection

###############################################################################
# 5. Resource Loading at Startup
###############################################################################
@app.on_event("startup")
async def load_resources():
    """
    Load all required resources:
      - Optimized emotion classification model
    """
    global emotion_model, emotion_tokenizer

    logger.info("Loading resources...")

    try:
        # 1) Load the tokenizer
        emotion_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

        # 2) Load the emotion classification model
        emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "bhadresh-savani/distilbert-base-uncased-emotion"
        )

        # 3) Optional GPU usage
        if torch.cuda.is_available():
            emotion_model.to("cuda")
            logger.info("Using GPU for inference.")
        else:
            logger.info("Using CPU for inference.")

        logger.info("All resources loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise RuntimeError(f"Failed to load resources: {str(e)}")

###############################################################################
# 6. Cached Emotion Detection
###############################################################################
@lru_cache(maxsize=100)
def cached_emotion_detection(english_text: str):
    """
    Perform tokenization and emotion classification on the given English text.
    Returns a probability distribution for each label:
      [p_anger, p_fear, p_joy, p_love, p_sadness, p_surprise]
    """
    try:
        inputs = emotion_tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = emotion_model(**inputs)
        probs = outputs.logits.softmax(dim=-1).cpu().numpy()[0]
        return probs
    except Exception as e:
        logger.error(f"Error during classification inference: {e}")
        raise RuntimeError(f"Error during classification inference: {str(e)}")

###############################################################################
# 7. Main Endpoint: Anxiety Analysis
###############################################################################
@app.post("/analyze/anxiety")
async def analyze_anxiety(input_data: dict):
    """
    Analyzes the input text (in English) for "anxiety."
    We interpret the 'fear' label as anxiety and check if it exceeds a threshold.

    Example payload:
    {
      "text": "I feel so worried and anxious about tomorrow."
    }

    Returns:
    {
      "is_anxious": bool,
      "anxiety_score": float,
      "emotion_distribution": [float, ... 6 items],
      "timestamp": ISO string,
      "language": "en"
    }
    """
    async with SEMAPHORE:
        if "text" not in input_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid input: 'text' is required."
            )
        raw_text = input_data["text"].strip()
        if not raw_text:
            raise HTTPException(
                status_code=400,
                detail="Invalid input: text cannot be empty."
            )

        # Run the emotion detection model
        probs = cached_emotion_detection(raw_text)

        # 'fear' => anxiety_score
        if "fear" in LABELS:
            fear_idx = LABELS.index("fear")
            anxiety_score = float(probs[fear_idx])
        else:
            anxiety_score = 0.0

        is_anxious = (anxiety_score > ANXIETY_THRESHOLD)

        return {
            "is_anxious": is_anxious,
            "anxiety_score": anxiety_score,
            "emotion_distribution": probs.tolist(),
            "timestamp": datetime.now().isoformat(),
            "language": "en"
        }
