import re
import pandas as pd
import joblib
import spacy
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("models/lgbm_model.pkl")
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace('.', '. ')
    text = text.replace('no experience', 'zero experience')
    doc = nlp(text)
    cleaned_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

def calculate_uppercase_ratio(text: str) -> float:
    text = str(text)
    words = text.split()
    if words:
        uppercase_words = sum(1 for word in words if word.isupper() and re.search('[A-Z]', word))
        ratio = uppercase_words / len(words)
    else:
        ratio = 0
    return ratio

def count_uppercase_words(text: str) -> int:
    text = str(text)
    words = text.split()
    if words:
        uppercase_words = sum(1 for word in words if word.isupper() and re.search('[A-Z]', word))
    else:
        uppercase_words = 0
    return uppercase_words

def calculate_digit_ratio(text: str) -> float:
    text = str(text)
    if text:
        num_digits = sum(c.isdigit() for c in text)
        digit_ratio = num_digits / len(text)
    else:
        digit_ratio = 0
    return digit_ratio

def calculate_special_char_ratio(text) -> float:
    text = str(text)
    if text:
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / len(text)
    else:
        special_char_ratio = 0
    return special_char_ratio

def count_scam_words(text: str) -> int:
    text = str(text)
    scam_words = [
        'free time', 'cash', 'today', 'no experience', 'zero experience',
        'from home', 'day', 'daily', 'extra', 'urgent', 'anytime', 'easy', 'easily'
    ]
    n_scam_words = sum(text.lower().count(word) for word in scam_words)
    return n_scam_words

class InputData(BaseModel):
    description: str

@app.post("/predict")
def predict(input_data: InputData, threshold: float = 0.3):
    description = input_data.description
    
    # New features
    char_count = len(description)
    word_count = len(description.split())
    uppercase_ratio = calculate_uppercase_ratio(description)
    n_uppercase_words = count_uppercase_words(description)
    digit_ratio = calculate_digit_ratio(description)
    special_char_ratio = calculate_special_char_ratio(description)
    scam_words = count_scam_words(description)
    description_cleaned = preprocess_text(description)
    
    # Create dataframe
    data = pd.DataFrame([{
        'description_cleaned': description_cleaned,
        'char_count': char_count,
        'uppercase_ratio': uppercase_ratio,
        'digit_ratio': digit_ratio,
        'special_char_ratio': special_char_ratio,
        'scam_words': scam_words,
        'n_uppercase_words': n_uppercase_words,
    }])

    # Prediction
    prob = model.predict_proba(data)[0][1]
    fraudulent = int(prob >= threshold)
    
    return {"fraudulent": fraudulent}