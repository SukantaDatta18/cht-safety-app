# ============================================================
# models.py - Loads models and makes predictions
# ============================================================

import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import pandas as pd
import json
import re
from fuzzywuzzy import fuzz
import os

# ---------- 1. SETUP DEVICE (CPU or GPU) ----------
# Check if your computer has a GPU. If not, it uses CPU (slower but works).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- 2. LOAD TOKENIZER ----------
# The tokenizer converts text into numbers that the model can understand.
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# ---------- 3. DEFINE THE MODEL STRUCTURE ----------
# This must match exactly how your model was trained in Colab.
class XLMRWithMCDropout(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

# ---------- 4. LOAD THE BINARY MODEL ----------
def load_binary_model(path='models/best_binary.pt'):
    print("Loading binary model...")
    model = XLMRWithMCDropout(num_labels=2).to(device)
    state_dict = torch.load(path, map_location=device)
    # Remove 'model.' prefix if it exists (common in saved models)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.model.load_state_dict(new_state_dict)
    model.eval()  # Set to evaluation mode (no dropout during predictions)
    print("✅ Binary model loaded successfully!")
    return model

# ---------- 5. LOAD THE MULTI-CLASS MODEL ----------
def load_multi_model(path='models/best_multi.pt'):
    print("Loading multi-class model...")
    model = XLMRWithMCDropout(num_labels=5).to(device)
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ Multi-class model loaded successfully!")
    return model

# ---------- 6. LOAD DATA FILES ----------
def load_gazetteer(path='data/location_gazetteer.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)  # Returns a list of all 329 location names

def load_coordinates(path='data/cht_coordinates.csv'):
    df = pd.read_csv(path)
    # Convert to dictionary: { 'place_name': {'lat': x, 'lon': y} }
    return df.set_index('name').to_dict(orient='index')

def load_severity_weights(path='data/severity_weights.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------- 7. EXTRACT LOCATIONS USING GAZETTEER ----------
def extract_gazetteer_locations(text, known_list, threshold=85):
    """
    Scan the text and find any location names from the gazetteer.
    Uses exact match first, then fuzzy partial matching (>=85% similarity).
    """
    found = []
    for loc in known_list:
        # Exact match check
        if loc in text:
            found.append(loc)
        # Fuzzy partial match
        elif fuzz.partial_ratio(loc, text) >= threshold:
            found.append(loc)
    
    # Remove duplicates while keeping order
    seen = set()
    unique = []
    for loc in found:
        if loc not in seen:
            seen.add(loc)
            unique.append(loc)
    return unique

# ---------- 8. GET COORDINATES FOR A LOCATION ----------
def get_coordinates(location_name, coord_dict):
    """
    Find the latitude and longitude for a given location.
    First tries exact match, then fuzzy match against the coordinates dictionary.
    """
    # Exact match
    if location_name in coord_dict:
        lat = coord_dict[location_name]['lat']
        lon = coord_dict[location_name]['lon']
        return lat, lon
    
    # Fuzzy match
    best_match = None
    best_score = 0
    for known_name in coord_dict.keys():
        score = fuzz.ratio(location_name, known_name)
        if score > best_score and score >= 80:
            best_score = score
            best_match = known_name
    
    if best_match:
        lat = coord_dict[best_match]['lat']
        lon = coord_dict[best_match]['lon']
        return lat, lon
    
    return None, None

# ---------- 9. CALCULATE RISK SCORE ----------
def calculate_risk(accident_type, severity_weights):
    """
    Calculate risk score based on accident type.
    Returns a score from 0-100.
    """
    # Debug: Print what we're looking for
    print(f"🔍 Looking for accident type: '{accident_type}'")
    print(f"📋 Available weights: {list(severity_weights.keys())}")
    
    # Try exact match first
    weight = severity_weights.get(accident_type)
    
    # If not found, try case-insensitive match
    if weight is None:
        for key in severity_weights.keys():
            if key.lower() == accident_type.lower():
                weight = severity_weights[key]
                print(f"✅ Found case-insensitive match: '{key}' -> {weight}")
                break
    
    # If still not found, use default
    if weight is None:
        print(f"⚠️ No weight found for '{accident_type}', using default 0.5")
        weight = 0.5
    
    score = int(weight * 100)
    print(f"📊 Risk score: {score}")
    return score

# ---------- 10. MAIN PREDICTION FUNCTION ----------
def predict(text, binary_model, multi_model, tokenizer, known_list, coord_dict, severity_weights):
    """
    Takes a text input and returns a dictionary with:
    - Whether it's an accident or not
    - The accident type (if applicable)
    - Confidence scores
    - Extracted locations
    - Coordinates for the first location
    - Risk score and level
    """
    
    # Clean text (remove extra spaces, etc.)
    text = text.strip()
    
    # Tokenize the text (convert to numbers for the model)
    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    
    # ---------- BINARY PREDICTION ----------
    with torch.no_grad():
        bin_logits = binary_model(input_ids, attention_mask)
        bin_probs = torch.softmax(bin_logits, dim=1).cpu().numpy()[0]
        bin_pred = int(torch.argmax(bin_logits, dim=1).cpu().numpy()[0])
    
    # Map numeric label to text
    binary_labels = {0: 'Non-Accident', 1: 'Accident'}
    is_accident = (bin_pred == 1)
    binary_label = binary_labels[bin_pred]
    binary_confidence = float(bin_probs[bin_pred])
    
    # ---------- MULTI-CLASS PREDICTION (ONLY IF ACCIDENT) ----------
    accident_type = None
    type_confidence = None
    risk_score = 0
    risk_level = 'None'
    
    if is_accident:
        with torch.no_grad():
            multi_logits = multi_model(input_ids, attention_mask)
            multi_probs = torch.softmax(multi_logits, dim=1).cpu().numpy()[0]
            multi_pred = int(torch.argmax(multi_logits, dim=1).cpu().numpy()[0])
        
        # Map numeric label to accident type
        multi_labels = {
            0: 'Road accident',
            1: 'Crime',
            2: 'Natural disaster',
            3: 'Waterway',
            4: 'Wildlife attack'
        }
        accident_type = multi_labels[multi_pred]
        type_confidence = float(multi_probs[multi_pred])
        
        # Calculate risk score
        risk_score = calculate_risk(accident_type, severity_weights)
        
        # Determine risk level based on score
        if risk_score >= 80:
            risk_level = 'Very High'
        elif risk_score >= 60:
            risk_level = 'High'
        elif risk_score >= 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
    
    # ---------- LOCATION EXTRACTION ----------
    locations = extract_gazetteer_locations(text, known_list, threshold=85)
    
    # ---------- COORDINATES ----------
    coordinates = None
    if locations:
        lat, lon = get_coordinates(locations[0], coord_dict)
        if lat is not None:
            coordinates = {'lat': lat, 'lon': lon}
    
    # ---------- BUILD THE RESULT ----------
    result = {
        'text': text,
        'is_accident': is_accident,
        'binary_label': binary_label,
        'binary_confidence': binary_confidence,
        'accident_type': accident_type,
        'type_confidence': type_confidence,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'locations': locations,
        'coordinates': coordinates
    }
    
    return result