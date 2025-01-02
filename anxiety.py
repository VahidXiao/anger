from fastapi import FastAPI, HTTPException
from datetime import datetime
import asyncio
import logging

from langdetect import detect
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

###############################################################################
# FastAPI Initialization and Configuration
###############################################################################
app = FastAPI(
    title="Multilingual Anxiety Detection",
    description="A multilingual anxiety detection service with translation and emotion analysis.",
    version="2.2.0",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

SEMAPHORE = asyncio.Semaphore(10)

LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
ANXIETY_THRESHOLD = 0.5  # Probability threshold for "anxious"

###############################################################################
# Load Models and Tokenizers
###############################################################################
@app.on_event("startup")
async def load_resources():
    global emotion_tokenizer, emotion_model, translator_models

    logger.info("Loading models...")
    try:
        emotion_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "bhadresh-savani/distilbert-base-uncased-emotion"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        translator_models = {
            "zh": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda" if torch.cuda.is_available() else "cpu"),
            "ja": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en").to("cuda" if torch.cuda.is_available() else "cpu"),
        }

        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise RuntimeError("Initialization error.")

###############################################################################
# Helper Functions
###############################################################################
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

def translate_to_en(text: str, lang: str) -> str:
    try:
        if lang in translator_models:
            tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            outputs = translator_models[lang].generate(**inputs)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        logger.warning(f"Translation failed for {lang}: {e}. Returning original text.")
        return text

def classify_emotions(text: str) -> torch.Tensor:
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        return logits.softmax(dim=-1).cpu().squeeze()
    except Exception as e:
        logger.error(f"Emotion classification failed: {e}")
        raise HTTPException(status_code=500, detail="Emotion classification error.")

###############################################################################
# API Endpoints
###############################################################################
@app.post("/analyze/anxiety")
async def analyze_anxiety(input_data: dict):
    async with SEMAPHORE:
        text = input_data.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Input 'text' cannot be empty.")

        lang = detect_language(text)
        english_text = translate_to_en(text, lang)
        emotion_probs = classify_emotions(english_text)

        anxiety_score = emotion_probs[LABELS.index("fear")]
        return {
            "is_anxious": anxiety_score > ANXIETY_THRESHOLD,
            "anxiety_score": float(anxiety_score),
            "emotion_distribution": emotion_probs.tolist(),
            "language": lang,
            "timestamp": datetime.utcnow().isoformat(),
        }