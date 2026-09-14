"""
core/dl_models.py — High-Performance Deep Learning Engine (NumPy & PyTorch)
Executes neural inference for Career ANN, MNIST Digit CNN, and Sentiment RNN/LSTM
without requiring TensorFlow or C++ compiler dependencies.
"""

import os
import io
import re
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


import numpy as np
import h5py
import torch
import torch.nn.functional as F

ANN_SKILLS = [
    "Python",
    "SQL",
    "AWS",
    "Docker",
    "Linux",
    "Git",
    "PowerBI",
    "Excel",
]

_CACHED_WEIGHTS = {}


def _get_model_path(filename: str, model_dir: str = None) -> Path:
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / "models"
    else:
        model_dir = Path(model_dir)
    return model_dir / filename


def _extract_h5_file(keras_path: Path) -> h5py.File:
    """Extracts in-memory h5py file from a .keras zip archive."""
    with zipfile.ZipFile(keras_path) as z:
        h5_bytes = io.BytesIO(z.read("model.weights.h5"))
        return h5py.File(h5_bytes, "r")


# ============================================================
# 1. ⚡ CAREER ANN (Artificial Neural Network)
# ============================================================

def load_ann_weights(model_dir: str = None):
    """Loads and caches weight matrices and class labels for Career ANN."""
    if "ann" in _CACHED_WEIGHTS:
        return _CACHED_WEIGHTS["ann"]

    ann_path = _get_model_path("career_ann.keras", model_dir)
    classes_path = _get_model_path("career_classes.npy", model_dir)

    h5 = _extract_h5_file(ann_path)
    w1 = h5["layers/dense/vars/0"][:]
    b1 = h5["layers/dense/vars/1"][:]
    w2 = h5["layers/dense_1/vars/0"][:]
    b2 = h5["layers/dense_1/vars/1"][:]
    w3 = h5["layers/dense_2/vars/0"][:]
    b3 = h5["layers/dense_2/vars/1"][:]
    classes = np.load(str(classes_path), allow_pickle=True)

    weights = {
        "w1": w1, "b1": b1,
        "w2": w2, "b2": b2,
        "w3": w3, "b3": b3,
        "classes": classes,
    }
    _CACHED_WEIGHTS["ann"] = weights
    return weights


def predict_career_ann(skill_vector: list, model_dir: str = None) -> dict:
    """
    Predicts career from 8 binary skill indicators using Dense ANN architecture:
    Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(5, Softmax)
    """
    weights = load_ann_weights(model_dir)
    x = np.array(skill_vector, dtype=np.float32)

    # Layer 1: Dense(8 -> 16, ReLU)
    h1 = np.maximum(0, np.dot(x, weights["w1"]) + weights["b1"])
    # Layer 2: Dense(16 -> 8, ReLU)
    h2 = np.maximum(0, np.dot(h1, weights["w2"]) + weights["b2"])
    # Layer 3: Dense(8 -> 5, Softmax)
    logits = np.dot(h2, weights["w3"]) + weights["b3"]
    exp_l = np.exp(logits - np.max(logits))
    probabilities = exp_l / np.sum(exp_l)

    classes = weights["classes"]
    top_idx = int(np.argmax(probabilities))
    top_career = str(classes[top_idx])
    confidence = float(probabilities[top_idx] * 100)

    breakdown = []
    for cls_name, prob in zip(classes, probabilities):
        breakdown.append({
            "career": str(cls_name),
            "probability": float(prob * 100)
        })
    breakdown = sorted(breakdown, key=lambda x: x["probability"], reverse=True)

    return {
        "top_career": top_career,
        "confidence": confidence,
        "breakdown": breakdown
    }


# ============================================================
# 2. ✍️ MNIST DIGIT CNN (Convolutional Neural Network)
# ============================================================

def load_cnn_weights(model_dir: str = None):
    """Loads and caches tensors for MNIST 2D CNN."""
    if "cnn" in _CACHED_WEIGHTS:
        return _CACHED_WEIGHTS["cnn"]

    cnn_path = _get_model_path("mnist_cnn.keras", model_dir)
    h5 = _extract_h5_file(cnn_path)

    w_c1 = h5["layers/conv2d/vars/0"][:]      # (3, 3, 1, 32)
    b_c1 = h5["layers/conv2d/vars/1"][:]      # (32,)
    w_c2 = h5["layers/conv2d_1/vars/0"][:]    # (3, 3, 32, 64)
    b_c2 = h5["layers/conv2d_1/vars/1"][:]    # (64,)
    w_d1 = h5["layers/dense/vars/0"][:]       # (1600, 64)
    b_d1 = h5["layers/dense/vars/1"][:]       # (64,)
    w_d2 = h5["layers/dense_1/vars/0"][:]     # (64, 10)
    b_d2 = h5["layers/dense_1/vars/1"][:]     # (10,)

    # Convert to PyTorch tensor format: (out_channels, in_channels, H, W)
    tensors = {
        "w_c1": torch.tensor(w_c1, dtype=torch.float32).permute(3, 2, 0, 1),
        "b_c1": torch.tensor(b_c1, dtype=torch.float32),
        "w_c2": torch.tensor(w_c2, dtype=torch.float32).permute(3, 2, 0, 1),
        "b_c2": torch.tensor(b_c2, dtype=torch.float32),
        "w_d1": torch.tensor(w_d1, dtype=torch.float32).t(),
        "b_d1": torch.tensor(b_d1, dtype=torch.float32),
        "w_d2": torch.tensor(w_d2, dtype=torch.float32).t(),
        "b_d2": torch.tensor(b_d2, dtype=torch.float32),
    }
    _CACHED_WEIGHTS["cnn"] = tensors
    return tensors


def predict_digit_cnn(image_28x28: np.ndarray, model_dir: str = None) -> dict:
    """
    Predicts digit from 28x28 normalized image array (values 0.0 - 1.0) using CNN:
    Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(64) -> Dense(10, Softmax)
    """
    tensors = load_cnn_weights(model_dir)

    if image_28x28.ndim == 2:
        img_t = torch.tensor(image_28x28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif image_28x28.ndim == 3:
        img_t = torch.tensor(image_28x28, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    else:
        img_t = torch.tensor(image_28x28, dtype=torch.float32)

    with torch.no_grad():
        # Conv 1 + Pool 1
        x = F.relu(F.conv2d(img_t, tensors["w_c1"], tensors["b_c1"]))
        x = F.max_pool2d(x, 2, 2)
        # Conv 2 + Pool 2
        x = F.relu(F.conv2d(x, tensors["w_c2"], tensors["b_c2"]))
        x = F.max_pool2d(x, 2, 2)
        # Flatten (NHWC format matching Keras)
        x = x.permute(0, 2, 3, 1).contiguous().view(1, -1)
        # Dense 1 + Dense 2
        x = F.relu(F.linear(x, tensors["w_d1"], tensors["b_d1"]))
        logits = F.linear(x, tensors["w_d2"], tensors["b_d2"])
        probs = F.softmax(logits, dim=-1).squeeze().numpy()

    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit] * 100)

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p * 100) for p in probs]
    }


# ============================================================
# 3. 💬 IMDB SENTIMENT (SimpleRNN vs. LSTM)
# ============================================================

_IMDB_WORD_INDEX = None


def _get_imdb_word_index() -> Dict[str, int]:
    """Loads IMDB word index map from data/ directory or creates standard vocabulary."""
    global _IMDB_WORD_INDEX
    if _IMDB_WORD_INDEX is not None:
        return _IMDB_WORD_INDEX

    json_path = Path(__file__).resolve().parent.parent / "data" / "imdb_word_index.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
            word_index = {w: idx + 3 for w, idx in raw_dict.items()}
            word_index["<PAD>"] = 0
            word_index["<START>"] = 1
            word_index["<UNK>"] = 2
            _IMDB_WORD_INDEX = word_index
            return _IMDB_WORD_INDEX

    # Fallback vocabulary
    _IMDB_WORD_INDEX = {"<PAD>": 0, "<START>": 1, "<UNK>": 2}
    return _IMDB_WORD_INDEX


def encode_sentiment_text(text: str, max_length: int = 200, vocab_size: int = 10000) -> List[int]:
    """Encodes and left-pads tokens to max_length matching Keras pad_sequences."""
    word_index = _get_imdb_word_index()
    words = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").split()
    
    encoded = [1]  # <START>
    for w in words:
        val = word_index.get(w, 2)
        if val >= vocab_size:
            val = 2
        encoded.append(val)
    
    # Left padding matching standard Keras sequence preprocessing
    if len(encoded) < max_length:
        padded = [0] * (max_length - len(encoded)) + encoded
    else:
        padded = encoded[:max_length]
    return padded


def load_rnn_weights(model_dir: str = None):
    """Loads SimpleRNN weights from keras file."""
    if "rnn" in _CACHED_WEIGHTS:
        return _CACHED_WEIGHTS["rnn"]

    rnn_path = _get_model_path("imdb_rnn.keras", model_dir)
    h5 = _extract_h5_file(rnn_path)

    weights = {
        "w_emb": h5["layers/embedding/vars/0"][:],
        "w_kernel": h5["layers/simple_rnn/cell/vars/0"][:],
        "w_recurrent": h5["layers/simple_rnn/cell/vars/1"][:],
        "b_rnn": h5["layers/simple_rnn/cell/vars/2"][:],
        "w_dense": h5["layers/dense/vars/0"][:],
        "b_dense": h5["layers/dense/vars/1"][:],
    }
    _CACHED_WEIGHTS["rnn"] = weights
    return weights


def load_lstm_weights(model_dir: str = None):
    """Loads LSTM weights from keras file."""
    if "lstm" in _CACHED_WEIGHTS:
        return _CACHED_WEIGHTS["lstm"]

    lstm_path = _get_model_path("imdb_lstm.keras", model_dir)
    h5 = _extract_h5_file(lstm_path)

    weights = {
        "w_emb": h5["layers/embedding/vars/0"][:],
        "w_kernel": h5["layers/lstm/cell/vars/0"][:],
        "w_recurrent": h5["layers/lstm/cell/vars/1"][:],
        "b_lstm": h5["layers/lstm/cell/vars/2"][:],
        "w_dense": h5["layers/dense/vars/0"][:],
        "b_dense": h5["layers/dense/vars/1"][:],
    }
    _CACHED_WEIGHTS["lstm"] = weights
    return weights


def predict_sentiment_rnn(text: str, model_dir: str = None) -> dict:
    """Executes SimpleRNN sequential forward pass."""
    w = load_rnn_weights(model_dir)
    tokens = encode_sentiment_text(text)

    h = np.zeros((32,), dtype=np.float32)
    for tok in tokens:
        x_t = w["w_emb"][tok]
        h = np.tanh(np.dot(x_t, w["w_kernel"]) + np.dot(h, w["w_recurrent"]) + w["b_rnn"])

    logit = np.dot(h, w["w_dense"])[0] + w["b_dense"][0]
    prob = float(1.0 / (1.0 + np.exp(-logit)))

    sentiment = "Positive / Confident" if prob >= 0.5 else "Passive / Hesitant"
    emoji = "🌟" if prob >= 0.5 else "⚠️"
    confidence = (prob if prob >= 0.5 else (1.0 - prob)) * 100

    return {
        "model_type": "SimpleRNN",
        "sentiment": sentiment,
        "emoji": emoji,
        "raw_score": prob,
        "confidence": confidence
    }


def predict_sentiment_lstm(text: str, model_dir: str = None) -> dict:
    """Executes LSTM sequential forward pass with long-term gating."""
    w = load_lstm_weights(model_dir)
    tokens = encode_sentiment_text(text)
    units = 32

    h = np.zeros((units,), dtype=np.float32)
    c = np.zeros((units,), dtype=np.float32)

    def _sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))

    for tok in tokens:
        x_t = w["w_emb"][tok]
        z = np.dot(x_t, w["w_kernel"]) + np.dot(h, w["w_recurrent"]) + w["b_lstm"]
        i = _sigmoid(z[0:units])
        f = _sigmoid(z[units:2*units])
        c_cand = np.tanh(z[2*units:3*units])
        o = _sigmoid(z[3*units:4*units])
        c = f * c + i * c_cand
        h = o * np.tanh(c)

    logit = np.dot(h, w["w_dense"])[0] + w["b_dense"][0]
    prob = float(_sigmoid(logit))

    sentiment = "Positive / Confident" if prob >= 0.5 else "Passive / Hesitant"
    emoji = "🌟" if prob >= 0.5 else "⚠️"
    confidence = (prob if prob >= 0.5 else (1.0 - prob)) * 100

    return {
        "model_type": "LSTM",
        "sentiment": sentiment,
        "emoji": emoji,
        "raw_score": prob,
        "confidence": confidence
    }


def compare_rnn_and_lstm(text: str, model_dir: str = None) -> dict:
    """Executes side-by-side inference on RNN and LSTM."""
    rnn_res = predict_sentiment_rnn(text, model_dir)
    lstm_res = predict_sentiment_lstm(text, model_dir)
    return {
        "rnn": rnn_res,
        "lstm": lstm_res,
        "input_text": text
    }


# ============================================================
# 4. 🎤 AI INTERVIEW & COVER LETTER TONE COACH
# ============================================================

WEAK_PHRASES_MAP = {
    "i think": "I demonstrated / I evaluated",
    "i feel like": "My analysis confirmed",
    "i tried to": "I spearheaded / I executed",
    "was responsible for": "Orchestrated / Managed",
    "sort of": "specifically",
    "kind of": "deliberately",
    "basically": "specifically",
    "just": "focused on",
    "only know a little": "possess foundational working knowledge in",
    "not really good at": "actively expanding proficiency in",
    "i guess": "evidence indicated",
}

STRONG_ACTION_VERBS = [
    "spearheaded", "engineered", "orchestrated", "architected", "optimized",
    "delivered", "accelerated", "implemented", "resolved", "collaborated",
    "streamlined", "automated", "designed", "scaled", "maximized"
]


def analyze_interview_tone_coach(text: str, model_dir: str = None) -> dict:
    """
    Evaluates interview responses and cover letters using LSTM recurrent sentiment,
    confidence metrics, weak word diagnostics, and STAR structure alignment.
    """
    # 1. Neural Recurrent Analysis
    recurrent_eval = compare_rnn_and_lstm(text, model_dir)
    lstm_score = recurrent_eval["lstm"]["raw_score"]
    rnn_score = recurrent_eval["rnn"]["raw_score"]

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    # 2. Action Verb & Weak Phrase Detection
    found_actions = [v for v in STRONG_ACTION_VERBS if v in text_lower]
    found_weak = []
    for weak_p, suggestion in WEAK_PHRASES_MAP.items():
        if weak_p in text_lower:
            found_weak.append({"phrase": weak_p, "fix": suggestion})

    # 3. STAR Framework Alignment
    star_breakdown = {
        "Situation": any(w in text_lower for w in ["when", "during", "project", "scenario", "context", "faced", "company", "team"]),
        "Task": any(w in text_lower for w in ["goal", "challenge", "objective", "tasked", "needed", "requirement", "problem"]),
        "Action": len(found_actions) > 0 or any(w in text_lower for w in ["implemented", "built", "created", "analyzed", "developed"]),
        "Result": any(w in text_lower for w in ["result", "achieved", "improved", "reduced", "increased", "%", "percent", "delivered", "saved"])
    }
    star_score = sum(1 for v in star_breakdown.values() if v)

    # 4. Overall Impact Score (0 - 100)
    star_points = (star_score / 4.0) * 40.0             # Up to 40 pts
    action_points = min(30.0, len(found_actions) * 10.0) # Up to 30 pts
    recurrent_points = min(20.0, max(5.0, (lstm_score + rnn_score) * 15.0)) # Up to 20 pts
    length_points = 10.0 if 20 <= word_count <= 300 else 5.0 # Up to 10 pts
    weak_penalty = min(40.0, len(found_weak) * 10.0)

    final_score = max(15.0, min(98.0, star_points + action_points + recurrent_points + length_points - weak_penalty))

    if final_score >= 80:
        tone_label = "Executive & High Impact"
        tone_badge = "🌟 Confident & Compelling"
        tone_color = "#10B981"
    elif final_score >= 55:
        tone_label = "Professional & Clear"
        tone_badge = "👍 Solid & Constructive"
        tone_color = "#6366F1"
    else:
        tone_label = "Passive / Needs Assertiveness"
        tone_badge = "⚠️ Hesitant / Needs Action Verbs"
        tone_color = "#F59E0B"

    # 5. Suggested Revision Draft

    rephrased_text = text
    for item in found_weak:
        pattern = re.compile(re.escape(item["phrase"]), re.IGNORECASE)
        rephrased_text = pattern.sub(f"**{item['fix']}**", rephrased_text)

    return {
        "impact_score": round(final_score, 1),
        "tone_label": tone_label,
        "tone_badge": tone_badge,
        "tone_color": tone_color,
        "word_count": word_count,
        "recurrent_eval": recurrent_eval,
        "found_action_verbs": found_actions,
        "found_weak_phrases": found_weak,
        "star_breakdown": star_breakdown,
        "star_score": star_score,
        "rephrased_preview": rephrased_text,
    }

