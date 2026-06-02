"""
Spendly AI — FastAPI Inference Server v2.0
===========================================
REST API untuk serving 4 model ML Spendly:
  - Receipt Detector : deteksi area struk dalam foto (CNN)
  - OCR CRNN         : ekstraksi teks dari gambar struk
  - Classifier       : klasifikasi 9 kategori pengeluaran (TF-IDF + CNN + SE)
  - Forecaster       : prediksi pengeluaran mingguan (LSTM + Attention)

Endpoint tambahan:
  - /insight          : saran keuangan via Google Gemini Generative AI
  - /process-receipt  : full pipeline (Detect → OCR → Classify)
  - /detect-receipt   : deteksi bounding box struk dalam foto
  - /health           : status server

Jalankan:
  uvicorn api.main:app --reload --port 8000

Coding Camp 2026 powered by DBS Foundation (CC26-PSU276)
"""

import os
import re
import sys
import string
import io
import logging
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from typing import List, Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as e:
    class _TensorFlowUnavailableLayer:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, *args, **kwargs):
            pass

        def get_config(self):
            return {}

    tf = SimpleNamespace(
        keras=SimpleNamespace(
            layers=SimpleNamespace(Layer=_TensorFlowUnavailableLayer),
            losses=SimpleNamespace(Loss=_TensorFlowUnavailableLayer),
        )
    )
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spendly-api")
if not TF_AVAILABLE:
    logger.warning(f"TensorFlow tidak tersedia. Model TF akan di-skip. Detail: {TF_IMPORT_ERROR}")

# ── Path Setup ────────────────────────────────────────────────────────────────
API_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# Add src/ to sys.path so SavedModel can find custom_components module
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Konstanta Global ──────────────────────────────────────────────────────────
# 9 kategori — Transport baru ditambahkan (belum ada data histori)
CATEGORIES = [
    "Beauty", "F&B", "Gas", "Groceries", "Health",
    "HouseHold", "Lifestyle", "Listrik", "Transport",
]
LABELS       = CATEGORIES
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES  = len(LABELS)            # 9
IMG_SIZE     = 224
LOOKBACK     = 12

## Forecaster juga dilatih pada 9 kategori (NB04 v2 — Transport ditambahkan)
# scaler.n_features_in_ = 9, Dense(9)
N_FORECAST_FEATURES = NUM_CLASSES  # 9

BLUR_THRESHOLD = 100

# OCR Charset — identik dengan NB05
CHARS_OCR     = string.digits + string.ascii_uppercase + string.ascii_lowercase + " .,:-/()%"
BLANK_IDX     = len(CHARS_OCR) + 1    # 72
NUM_OCR_CHARS = len(CHARS_OCR) + 2    # 73
OCR_IMG_W     = 400
OCR_IMG_H     = 50

EASYOCR_LANGS = [
    lang.strip()
    for lang in os.environ.get("EASYOCR_LANGS", "id,en").split(",")
    if lang.strip()
]
EASYOCR_GPU = os.environ.get("EASYOCR_GPU", "false").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
easyocr_reader = None


# ══════════════════════════════════════════════════════════════════════════════
# INLINE CUSTOM KERAS COMPONENTS (self-contained — no src/ imports)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sastrawi (optional) ──────────────────────────────────────────────────────
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    logger.warning("PySastrawi tidak tersedia. NLP preprocessing akan di-skip.")


class NLPPreprocessor:
    """Pipeline NLP untuk teks transaksi Bahasa Indonesia.
    Identik dengan definisi di NB03 & NB06 dan custom_components.py.
    """
    def __init__(self):
        if SASTRAWI_AVAILABLE:
            self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
            self.stemmer = StemmerFactory().create_stemmer()
        else:
            self.stopword_remover = None
            self.stemmer = None

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\d+", "NUM", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self.stopword_remover:
            text = self.stopword_remover.remove(text)
        if self.stemmer:
            text = self.stemmer.stem(text)
        return text

    def preprocess_batch(self, texts):
        return [self.preprocess(t) for t in texts]


# ── SEBlock (Squeeze-and-Excitation) — identik NB06 / custom_components.py ──
class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block untuk CNN branch classifier."""

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap    = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            max(1, channels // self.ratio), activation="relu", use_bias=False
        )
        self.dense2 = tf.keras.layers.Dense(
            channels, activation="sigmoid", use_bias=False
        )
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


# ── AttentionLayer — identik NB04 / custom_components.py ────────────────────
class AttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer untuk sequence modeling (LSTM forecaster).

    Input shape : (batch, timesteps, features)
    Output shape: (batch, features)
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))                  # (batch, T, 1)
        attention_weights = tf.nn.softmax(score, axis=1)            # (batch, T, 1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ── FocalLoss — identik custom_components.py (needed for .keras loading) ────
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss dengan per-class alpha untuk class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

        if class_weights is not None:
            vals = list(class_weights.values())
            max_w = max(vals)
            self._alpha_tensor = tf.constant(
                [class_weights.get(i, 1.0) / max_w for i in range(len(class_weights))],
                dtype=tf.float32,
            )
        elif isinstance(alpha, (list, tuple)):
            self._alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        else:
            self._alpha_tensor = float(alpha)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal  = tf.math.pow(1.0 - pt, self.gamma) * ce

        if isinstance(self._alpha_tensor, float):
            weighted = self._alpha_tensor * focal
        else:
            alpha_t  = tf.reshape(self._alpha_tensor, (1, -1))
            weighted = alpha_t * y_true * focal

        return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "class_weights": self.class_weights,
        })
        return config


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER — hanya forecaster yang perlu di-rebuild (menggunakan .h5)
# ══════════════════════════════════════════════════════════════════════════════

def build_forecaster(lookback: int = LOOKBACK, n_features: int = N_FORECAST_FEATURES):
    """Bangun arsitektur forecaster — identik dengan NB04 v2.

    LSTM(64, ret_seq) → LSTM(32, ret_seq) → AttentionLayer → Dense(32) → Dense(9)
    """
    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = AttentionLayer(name="attention")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs, name="spendly_forecaster_attention")


def _build_ocr_model():
    """Rebuild OCR CRNN architecture — identik dengan NB05.

    3×Conv+BN+MaxPool → SpatialDropout → Reshape → 2×BiLSTM → Dense → Dense(73)
    Hanya digunakan jika .keras file tidak kompatibel dengan Keras versi ini.
    """
    from tensorflow.keras import layers

    img_w, img_h = 400, 50
    num_chars = NUM_OCR_CHARS  # 73

    inputs = tf.keras.Input(shape=(img_h, img_w, 1), name="ocr_input")

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)       # → (25, 200, 32)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)       # → (12, 100, 64)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)       # → (6, 100, 128)

    # Conv Block 4 + SpatialDropout
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.3)(x)                 # → (6, 100, 128)

    # Reshape for RNN: (batch, time=100, features=768)
    x = layers.Permute((2, 1, 3))(x)                    # → (100, 6, 128)
    x = layers.Reshape((-1, 6 * 128))(x)                # → (100, 768)

    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Dense output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_chars, activation="linear", name="logits")(x)  # → (100, 73)

    return tf.keras.Model(inputs, outputs, name="spendly_ocr_crnn")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assess_blur(image_bgr: np.ndarray) -> dict:
    """Laplacian variance blur detection — identik NB01."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"blur_score": blur_score, "is_blurry": blur_score < BLUR_THRESHOLD}


def preprocess_image_classifier(image_bgr: np.ndarray) -> np.ndarray:
    """Load & resize untuk classifier — identik NB03."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0


def preprocess_image_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + Adaptive Threshold + Denoise — identik NB05."""
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    gray = cv2.resize(gray, (OCR_IMG_W, OCR_IMG_H))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return gray.astype(np.float32) / 255.0


def preprocess_image_detector(image_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize dan normalize gambar untuk receipt detector.

    Args:
        image_bgr: input BGR image
        target_size: (height, width) sesuai input shape model
    Returns:
        normalized float32 array (1, H, W, 3)
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))  # cv2 uses (W, H)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def decode_ocr(logits: np.ndarray) -> str:
    """CTC greedy decode — identik NB05."""
    logits = tf.cast(logits, tf.float32)
    pred_indices = tf.argmax(logits, axis=-1).numpy()[0]
    text = ""
    prev_idx = -1
    for idx in pred_indices:
        if idx != prev_idx:
            if 1 <= idx <= len(CHARS_OCR):
                text += CHARS_OCR[idx - 1]
        prev_idx = idx
    return text


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Konversi bytes upload ke BGR numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format JPG/PNG.")
    return img


def get_upload_file(file: Optional[UploadFile], receipt: Optional[UploadFile] = None) -> UploadFile:
    """Accept both backend-style `file` and frontend-style `receipt` form fields."""
    upload = file or receipt
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="File gambar wajib dikirim dengan field 'file' atau 'receipt'.",
        )
    return upload


def get_model_input_size(model, default: tuple = (224, 224)) -> tuple:
    """Return (height, width) for Keras models and safe defaults for SavedModel wrappers."""
    shape = getattr(model, "input_shape", None)
    if shape and len(shape) >= 3:
        height = shape[1] if shape[1] else default[0]
        width = shape[2] if shape[2] else default[1]
        return int(height), int(width)
    return default


def is_easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def get_easyocr_reader():
    """Lazy-load EasyOCR so API startup stays reliable and fast."""
    global easyocr_reader
    if easyocr_reader is not None:
        return easyocr_reader

    try:
        import easyocr
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"EasyOCR belum tersedia. Install dependency easyocr. Detail: {e}",
        )

    try:
        easyocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU)
    except Exception as first_error:
        if EASYOCR_LANGS == ["en"]:
            raise HTTPException(status_code=503, detail=f"EasyOCR gagal dimuat: {first_error}")
        logger.warning(f"EasyOCR gagal dengan bahasa {EASYOCR_LANGS}: {first_error}. Fallback ke ['en'].")
        easyocr_reader = easyocr.Reader(["en"], gpu=EASYOCR_GPU)
    return easyocr_reader


def _bbox_stats(bbox) -> dict:
    xs = [float(point[0]) for point in bbox]
    ys = [float(point[1]) for point in bbox]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
        "y_center": sum(ys) / len(ys),
    }


def run_easyocr(image_bgr: np.ndarray) -> dict:
    """Read the full receipt with EasyOCR and return sorted text lines."""
    reader = get_easyocr_reader()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw_results = reader.readtext(image_rgb, detail=1, paragraph=False)

    lines = []
    for result in raw_results:
        if len(result) < 3:
            continue
        bbox, text, confidence = result[0], str(result[1]).strip(), float(result[2])
        if not text:
            continue
        box = _bbox_stats(bbox)
        lines.append({
            "text": re.sub(r"\s+", " ", text).strip(),
            "confidence": confidence,
            "bbox": {
                "x_min": round(box["x_min"], 2),
                "y_min": round(box["y_min"], 2),
                "x_max": round(box["x_max"], 2),
                "y_max": round(box["y_max"], 2),
            },
            "_sort_y": box["y_center"],
            "_sort_x": box["x_min"],
        })

    lines.sort(key=lambda item: (round(item["_sort_y"] / 12), item["_sort_x"]))
    for line in lines:
        line.pop("_sort_y", None)
        line.pop("_sort_x", None)

    raw_text = "\n".join(line["text"] for line in lines)
    avg_confidence = float(np.mean([line["confidence"] for line in lines])) if lines else 0.0
    return {
        "raw_text": raw_text,
        "lines": lines,
        "avg_confidence": avg_confidence,
    }


MERCHANT_SKIP_WORDS = {
    "alamat", "amount", "bayar", "bill", "bukti", "cab", "cabang", "cashier",
    "date", "diskon", "email", "faktur", "invoice", "jalan", "jl", "jln", "kasir",
    "kembali", "member", "npwp", "nota", "order", "pajak", "phone", "ppn",
    "receipt", "shift", "struk", "subtotal", "tax", "tel", "telp", "tanggal",
    "time", "total", "transaction", "transaksi", "www",
}

MERCHANT_ALIASES = [
    (r"\bindomaret\b|\bindo\s*maret\b|\bindomarco\b", "Indomaret"),
    (r"\balfamart\b|\balfa\s*mart\b|\bsumber\s+alfaria\b", "Alfamart"),
    (r"\balfamidi\b|\balfa\s*midi\b", "Alfamidi"),
    (r"\bsuper\s*indo\b|\bsuperindo\b", "Super Indo"),
    (r"\bhypermart\b", "Hypermart"),
    (r"\btransmart\b", "Transmart"),
    (r"\blotte\s*mart\b|\blottemart\b", "Lotte Mart"),
    (r"\bfarmers?\s+market\b", "Farmers Market"),
    (r"\bstarbucks\b", "Starbucks"),
    (r"\bfore\s+coffee\b|\bfore\b", "Fore Coffee"),
    (r"\bkopi\s+kenangan\b", "Kopi Kenangan"),
    (r"\bjanji\s+jiwa\b", "Janji Jiwa"),
    (r"\bmcdonald'?s?\b|\bmc\s*donald\b|\bmcd\b", "McDonald's"),
    (r"\bkfc\b|\bkentucky\b", "KFC"),
    (r"\bsolaria\b", "Solaria"),
    (r"\bmixue\b", "Mixue"),
    (r"\bchatime\b", "Chatime"),
    (r"\bj\.?\s*co\b|\bjco\b", "J.CO"),
    (r"\bguardian\b", "Guardian"),
    (r"\bwatsons?\b", "Watsons"),
    (r"\bkimia\s+farma\b", "Kimia Farma"),
    (r"\bcentury\b", "Century"),
    (r"\bace\s+hardware\b", "ACE Hardware"),
    (r"\bmr\.?\s*diy\b|\bmrdiy\b", "MR.DIY"),
    (r"\bminiso\b", "Miniso"),
    (r"\buniqlo\b", "Uniqlo"),
    (r"\bgrab\b", "Grab"),
    (r"\bgojek\b|\bgo-jek\b", "Gojek"),
    (r"\bpln\b", "PLN"),
]

MONTHS_ID = {
    "jan": 1, "januari": 1, "january": 1,
    "feb": 2, "februari": 2, "february": 2,
    "mar": 3, "maret": 3, "march": 3,
    "apr": 4, "april": 4,
    "mei": 5, "may": 5,
    "jun": 6, "juni": 6, "june": 6,
    "jul": 7, "juli": 7, "july": 7,
    "agu": 8, "agustus": 8, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oct": 10, "oktober": 10, "october": 10,
    "nov": 11, "november": 11,
    "des": 12, "dec": 12, "desember": 12, "december": 12,
}


def _is_date_like(text: str) -> bool:
    return bool(
        re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)
        or re.search(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b", text)
    )


def _looks_like_amount(text: str) -> bool:
    return bool(re.search(r"\b(?:rp|idr)?\s*\d{1,3}(?:[.,]\d{3})+\b", text, re.I))


def clean_merchant_name(text: str) -> str:
    text = re.sub(r"(?i)\b(pt|cv|tbk|ltd)\.?\b", "", text)
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9)&.' -]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -:|")
    return text[:80]


def extract_merchant_name(lines: List[str]) -> tuple:
    search_area = "\n".join(lines[:14]).lower()
    for pattern, name in MERCHANT_ALIASES:
        if re.search(pattern, search_area, re.I):
            return name, 0.97

    candidates = []
    for index, line in enumerate(lines[:12]):
        clean = clean_merchant_name(line)
        lower = clean.lower()
        if not clean or len(clean) < 3:
            continue
        if not re.search(r"[A-Za-z]", clean):
            continue
        if re.search(r"(https?://|www\.|@)", lower):
            continue
        if _is_date_like(clean) or _looks_like_amount(clean):
            continue
        tokens = set(re.findall(r"[a-z]+", lower))
        if tokens & MERCHANT_SKIP_WORDS:
            continue
        digit_ratio = sum(char.isdigit() for char in clean) / max(len(clean), 1)
        alpha_ratio = sum(char.isalpha() for char in clean) / max(len(clean), 1)
        if digit_ratio > 0.25 or alpha_ratio < 0.45:
            continue
        if len(clean) > 45 and not any(word in lower for word in ["mart", "market", "coffee", "cafe", "resto", "apotek"]):
            continue

        score = 100 - (index * 9)
        if clean.isupper():
            score += 8
        if any(word in lower for word in ["mart", "market", "store", "coffee", "cafe", "resto", "apotek", "pharmacy"]):
            score += 14
        if any(word in lower for word in ["jl", "jalan", "lantai", "unit", "blok"]):
            score -= 35
        candidates.append((score, clean))

    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True, key=lambda item: item[0])
    confidence = min(0.95, max(0.55, candidates[0][0] / 120))
    return candidates[0][1], round(confidence, 4)


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year <= 50 else 1900 + year
    return year


def _safe_iso_date(day: int, month: int, year: int) -> Optional[str]:
    try:
        parsed = date(_normalize_year(year), month, day)
    except ValueError:
        return None
    max_future = date(date.today().year + 1, 12, 31)
    if parsed.year < 2000 or parsed > max_future:
        return None
    return parsed.isoformat()


def extract_scan_date(lines: List[str]) -> tuple:
    scored_matches = []
    month_pattern = "|".join(sorted(MONTHS_ID.keys(), key=len, reverse=True))

    for index, line in enumerate(lines):
        lower = line.lower()
        if any(word in lower for word in ["npwp", "tel", "phone", "invoice", "faktur", "struk", "receipt"]):
            penalty = 0.15
        else:
            penalty = 0.0

        for match in re.finditer(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b", line):
            iso = _safe_iso_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            if iso:
                scored_matches.append((0.95 - penalty - index * 0.005, iso))

        for match in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", line):
            first = int(match.group(1))
            second = int(match.group(2))
            year = int(match.group(3))
            variants = [(first, second)]
            if first <= 12 < second:
                variants.insert(0, (second, first))
            for day, month in variants:
                iso = _safe_iso_date(day, month, year)
                if iso:
                    scored_matches.append((0.9 - penalty - index * 0.005, iso))
                    break

        regex = re.compile(rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{2,4}})\b", re.I)
        for match in regex.finditer(line):
            month = MONTHS_ID.get(match.group(2).lower())
            if not month:
                continue
            iso = _safe_iso_date(int(match.group(1)), month, int(match.group(3)))
            if iso:
                scored_matches.append((0.9 - penalty - index * 0.005, iso))

    if scored_matches:
        scored_matches.sort(reverse=True, key=lambda item: item[0])
        return scored_matches[0][1], round(max(0.45, scored_matches[0][0]), 4)

    return None, 0.0


def _parse_idr_token(token: str) -> Optional[int]:
    token = token.upper()
    token = re.sub(r"\b(?:RP|IDR)\.?\b", "", token)
    token = token.replace("O", "0").replace("I", "1").replace("L", "1").replace("S", "5")
    token = re.sub(r"[^0-9.,\s]", "", token).strip()
    if not token:
        return None

    token = re.sub(r"\s+", ".", token)
    last_dot = token.rfind(".")
    last_comma = token.rfind(",")
    last_sep = max(last_dot, last_comma)
    integer_part = token
    if last_sep != -1:
        decimals = re.sub(r"\D", "", token[last_sep + 1:])
        if len(decimals) == 2:
            integer_part = token[:last_sep]

    digits = re.sub(r"\D", "", integer_part)
    if len(digits) < 3:
        return None
    value = int(digits)
    if value < 100 or value > 100_000_000:
        return None
    return value


def _amount_values_from_line(line: str) -> List[int]:
    normalized = line.upper()
    ocr_safe = normalized.replace("O", "0").replace("I", "1").replace("L", "1").replace("S", "5")
    patterns = [
        r"(?:RP|IDR)\.?\s*[0-9OILS]{1,3}(?:[.,\s][0-9OILS]{3})+(?:[.,][0-9OILS]{2})?",
        r"(?:RP|IDR)\.?\s*[0-9OILS]{4,9}(?:[.,][0-9OILS]{2})?",
        r"\b[0-9]{1,3}(?:[.,\s][0-9]{3})+(?:[.,][0-9]{2})?\b",
        r"\b[0-9]{4,9}\b",
    ]
    values = []
    for amount_text in {normalized, ocr_safe}:
        for pattern in patterns:
            for match in re.finditer(pattern, amount_text):
                value = _parse_idr_token(match.group(0))
                if value is not None:
                    values.append(value)
    return sorted(set(values))


def _has_phrase(text: str, phrases: List[str]) -> bool:
    normalized_text = text.lower().replace("0", "o").replace("1", "l")
    normalized_text = re.sub(r"[^a-z0-9&]+", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    for phrase in phrases:
        normalized_phrase = phrase.lower().replace("0", "o").replace("1", "l")
        normalized_phrase = re.sub(r"[^a-z0-9&]+", " ", normalized_phrase)
        words = [re.escape(part) for part in normalized_phrase.split()]
        if words and re.search(r"\b" + r"\s+".join(words) + r"\b", normalized_text):
            return True
    return False


def extract_total_amount(lines: List[str]) -> tuple:
    specific_total_phrases = [
        "grand total", "total harga", "harga total", "total hrg", "hrg total",
        "total price", "price total", "total amount", "amount total",
        "total bayar", "bayar total", "total dibayar", "dibayar total",
        "total pembayaran", "pembayaran total", "total belanja", "belanja total",
        "total pembelian", "pembelian total", "total penjualan", "penjualan total",
        "total tagihan", "tagihan total", "total transaksi", "total payment",
        "payment total", "total sales", "sales total", "net sales", "net total",
        "nett total", "netto total", "total nett", "total netto",
        "jumlah bayar", "jumlah dibayar", "jumlah belanja", "jumlah pembelian",
        "jumlah tagihan", "jml bayar", "jml dibayar", "jml belanja",
        "jml pembelian", "jml tagihan", "jlh bayar", "jlh belanja",
        "amount due", "amt due", "total due", "balance due", "harus dibayar",
        "yang harus dibayar", "total yang harus dibayar", "ttl harga",
        "ttl bayar", "ttl dibayar", "ttl belanja", "tot harga", "tot bayar",
        "tot belanja",
    ]
    generic_total_phrases = ["total", "ttl", "tot"]
    medium_phrases = [
        "harga", "hrg", "price", "amount", "jumlah", "jml", "jlh", "tagihan",
        "netto", "nett", "bayar", "dibayar", "belanja", "pembelian",
        "penjualan", "due",
    ]
    negative_phrases = [
        "subtotal", "sub total", "ppn", "pajak", "tax", "discount", "diskon",
        "disc", "promo", "hemat", "kembali", "kembalian", "change", "voucher",
        "saldo", "rounding", "pembulatan", "cashback", "point", "poin",
        "service charge", "service", "svc", "admin", "biaya admin", "dpp", "pb1",
    ]
    item_count_phrases = [
        "qty", "quantity", "kuantitas", "jumlah item", "total item", "item",
        "pcs", "pc", "x",
    ]
    noisy_words = [
        "invoice", "faktur", "struk", "receipt", "order", "phone", "tel",
        "telp", "npwp", "member", "cashier", "kasir", "shift", "auth",
        "approval", "reff", "ref",
    ]
    payment_words = [
        "cash", "tunai", "debit", "credit", "kredit", "kartu", "qris", "gopay",
        "ovo", "dana", "shopeepay", "flazz", "bca", "bri", "bni", "mandiri",
        "visa", "mastercard",
    ]
    payment_total_phrases = [
        "total bayar", "total dibayar", "jumlah bayar", "jumlah dibayar",
        "total pembayaran", "amount due", "amt due", "total due",
        "harus dibayar", "yang harus dibayar",
    ]
    blockers = negative_phrases + payment_words + item_count_phrases + noisy_words

    candidates = []
    fallback_values = []
    total_lines = len(lines)
    for index, line in enumerate(lines):
        lower = line.lower()
        next_line = lines[index + 1] if index + 1 < total_lines else ""
        previous_line = lines[index - 1] if index > 0 else ""

        has_specific_total_label = _has_phrase(lower, specific_total_phrases)
        has_generic_total_label = _has_phrase(lower, generic_total_phrases)
        has_strong_label = has_specific_total_label or has_generic_total_label
        has_medium_label = _has_phrase(lower, medium_phrases)
        has_negative_label = _has_phrase(lower, negative_phrases)
        has_item_count_label = _has_phrase(lower, item_count_phrases)
        has_payment_label = _has_phrase(lower, payment_words)
        has_noisy_label = _has_phrase(lower, noisy_words)

        values = _amount_values_from_line(line)
        amount_source = line
        used_next_line = False
        if not values and has_strong_label and next_line:
            next_lower = next_line.lower()
            next_has_blocker = _has_phrase(next_lower, blockers)
            if not next_has_blocker and not _is_date_like(next_line):
                values = _amount_values_from_line(next_line)
                amount_source = next_line
                used_next_line = bool(values)
        if not values:
            continue
        if _is_date_like(line) or _is_date_like(amount_source):
            continue

        for value in values:
            if has_noisy_label and not has_strong_label:
                continue

            score = 0.0
            if has_specific_total_label:
                score += 125
            elif has_generic_total_label:
                score += 95
            elif has_medium_label:
                score += 45
            if _has_phrase(previous_line.lower(), specific_total_phrases) and not _has_phrase(previous_line.lower(), blockers):
                score += 45
            elif _has_phrase(previous_line.lower(), generic_total_phrases) and not _has_phrase(previous_line.lower(), blockers):
                score += 35
            if has_negative_label:
                score -= 125
            if has_item_count_label:
                score -= 100
            if has_payment_label:
                payment_is_total = _has_phrase(lower, payment_total_phrases)
                score -= 25 if payment_is_total else 95
            if re.search(r"\b(?:rp|idr)\b", f"{line} {amount_source}".lower()):
                score += 10
            if used_next_line:
                score += 8
            score += min(12, index / max(total_lines, 1) * 12)
            score += min(8, value / 250_000)
            candidates.append((score, value))

            if not _has_phrase(lower, blockers) and not _has_phrase(amount_source.lower(), blockers):
                fallback_score = index / max(total_lines, 1) * 30
                if has_medium_label:
                    fallback_score += 15
                if re.search(r"\b(?:rp|idr)\b", amount_source.lower()):
                    fallback_score += 8
                fallback_score += min(8, value / 250_000)
                fallback_values.append((fallback_score, value))

    positive_candidates = [item for item in candidates if item[0] > 40]
    if positive_candidates:
        positive_candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        score, value = positive_candidates[0]
        return value, round(0.95 if score >= 90 else 0.78, 4)

    if fallback_values:
        fallback_values = [item for item in fallback_values if item[1] >= 1_000]
        if fallback_values:
            fallback_values.sort(reverse=True, key=lambda item: (item[0], item[1]))
            return fallback_values[0][1], 0.55

    return None, 0.0


def confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def classify_spending(image_bgr: np.ndarray, text: str, is_blurry: bool = False) -> dict:
    if not all(key in models for key in ["classifier", "nlp", "tfidf"]):
        return {
            "category": None,
            "confidence": 0.0,
            "all_scores": {},
            "text_used": "",
        }

    input_text = "" if is_blurry else text
    nlp = models["nlp"]
    tfidf = models["tfidf"]
    clean_text = nlp.preprocess(input_text)
    text_vector = tfidf.transform([clean_text]).toarray().astype(np.float32)

    img_arr = preprocess_image_classifier(image_bgr)
    img_tensor = np.expand_dims(img_arr, axis=0)

    if "classifier_infer" in models:
        infer_fn = models["classifier_infer"]
        result = infer_fn(
            text_input=tf.constant(text_vector, dtype=tf.float32),
            image_input=tf.constant(img_tensor, dtype=tf.float32),
        )
        preds = next(iter(result.values())).numpy()
    else:
        preds = model_predict(models["classifier"], [text_vector, img_tensor])

    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    return {
        "category": LABELS[pred_idx],
        "confidence": round(confidence, 4),
        "all_scores": {
            LABELS[i]: round(float(preds[0][i]), 4) for i in range(NUM_CLASSES)
        },
        "text_used": clean_text,
    }


def extract_receipt_fields(ocr_result: dict, classification: dict) -> dict:
    lines = [line["text"] for line in ocr_result["lines"]]
    merchant_name, merchant_conf = extract_merchant_name(lines)
    scan_date, date_conf = extract_scan_date(lines)
    total_amount, amount_conf = extract_total_amount(lines)

    category = classification.get("category")
    classifier_conf = float(classification.get("confidence") or 0.0)
    available_scores = [
        score for score in [merchant_conf, date_conf, amount_conf, classifier_conf]
        if score > 0
    ]
    confidence_score = round(float(np.mean(available_scores)) if available_scores else 0.0, 4)

    return {
        "merchant_name": merchant_name,
        "total_amount": total_amount,
        "scan_date": scan_date,
        "suggested_category_id": None,
        "suggested_category_name": category,
        "suggested_category_icon": None,
        "suggested_category_color": None,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level(confidence_score),
        "raw_text": ocr_result["raw_text"],
    }


def model_predict(model, inputs):
    """Universal prediction: works with Keras Model, TFSMLayer, or saved_model object.

    - Keras Model: uses model.predict()
    - TFSMLayer: calls model(inputs) — returns dict, extract first output
    - tf.saved_model: calls model.signatures['serving_default'](input_tensor)

    `inputs` can be a single array or a list of arrays (multi-input model).
    """
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow tidak tersedia: {TF_IMPORT_ERROR}")

    if hasattr(model, 'predict'):
        # Standard Keras Model
        return model.predict(inputs, verbose=0)
    elif isinstance(model, tf.keras.layers.Layer):
        # TFSMLayer — needs tensor inputs as keyword args
        if isinstance(inputs, (list, tuple)):
            # Multi-input model: convert each to tensor
            tensor_inputs = [
                tf.constant(x, dtype=tf.float32) if not isinstance(x, tf.Tensor) else x
                for x in inputs
            ]
            # TFSMLayer wraps a SavedModel — we need to call it with named kwargs
            # matching the input tensor names. Use tf.saved_model.load() to find them.
            endpoint_fn = None
            input_keys = None
            # Try to get input names from the layer's endpoint
            for attr in ('_callable', '_endpoint'):
                fn = getattr(model, attr, None)
                if fn is not None and hasattr(fn, 'structured_input_signature'):
                    input_keys = list(fn.structured_input_signature[1].keys())
                    break
            if input_keys is None:
                # Reload SavedModel to discover input names
                saved_path = getattr(model, '_asset_path', None) or getattr(model, 'filepath', None)
                if saved_path:
                    sm = tf.saved_model.load(str(saved_path))
                    sig = sm.signatures.get('serving_default')
                    if sig:
                        input_keys = list(sig.structured_input_signature[1].keys())
                        endpoint_fn = sig
            if input_keys and len(input_keys) >= len(tensor_inputs):
                kwargs = {input_keys[i]: tensor_inputs[i] for i in range(len(tensor_inputs))}
                if endpoint_fn is not None:
                    result = endpoint_fn(**kwargs)
                else:
                    result = model(**kwargs)
            else:
                # Last fallback: call with dict positional
                result = model(tensor_inputs[0], tensor_inputs[1] if len(tensor_inputs) > 1 else None)
        else:
            input_tensor = tf.constant(inputs, dtype=tf.float32) if not isinstance(inputs, tf.Tensor) else inputs
            result = model(input_tensor)

        if isinstance(result, dict):
            val = next(iter(result.values()))
            return val.numpy() if hasattr(val, 'numpy') else np.array(val)
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)
    elif hasattr(model, 'signatures'):
        # tf.saved_model object
        infer = model.signatures['serving_default']
        input_keys = list(infer.structured_input_signature[1].keys())
        if isinstance(inputs, (list, tuple)):
            kwargs = {input_keys[i]: tf.constant(inputs[i], dtype=tf.float32) for i in range(min(len(input_keys), len(inputs)))}
        else:
            kwargs = {input_keys[0]: tf.constant(inputs, dtype=tf.float32)}
        result = infer(**kwargs)
        val = next(iter(result.values()))
        return val.numpy()
    else:
        # Fallback
        if isinstance(inputs, (list, tuple)):
            tensors = [tf.constant(x, dtype=tf.float32) for x in inputs]
            result = model(*tensors)
        else:
            result = model(tf.constant(inputs, dtype=tf.float32))
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE & LOADING
# ══════════════════════════════════════════════════════════════════════════════
models = {}

# Custom objects dict untuk tf.keras.models.load_model()
CUSTOM_OBJECTS = {
    "SEBlock": SEBlock,
    "AttentionLayer": AttentionLayer,
    "FocalLoss": FocalLoss,
}


def load_all_models():
    """Load semua model saat startup.

    Strategi loading (kompatibel TF 2.10 dan TF 2.21/Keras 3):
      1. Coba tf.keras.models.load_model() pada .keras file
      2. Fallback ke tf.saved_model.load() pada SavedModel dir
      3. Fallback ke rebuild architecture + load .h5 weights
    """
    logger.info("=" * 60)
    logger.info("Loading Spendly AI models...")
    if not TF_AVAILABLE:
        logger.warning(f"TensorFlow unavailable; skipping TensorFlow models. Detail: {TF_IMPORT_ERROR}")
        logger.info("EasyOCR will be loaded lazily on the first OCR request.")
        logger.info("=" * 60)
        return

    logger.info(f"TensorFlow {tf.__version__}")
    logger.info("=" * 60)

    def _try_load_keras_or_savedmodel(keras_path, savedmodel_path, name, custom_objects=None):
        """Try .keras first, then SavedModel, return (model, method_str) or raise."""
        # 1. Try .keras (works on TF 2.10-style .keras = HDF5, or Keras 3 ZIP)
        if os.path.exists(keras_path):
            try:
                m = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
                return m, ".keras"
            except Exception as e1:
                logger.warning(f"  {name} .keras load failed: {e1}")

        # 2. Try SavedModel directory
        if savedmodel_path and os.path.exists(savedmodel_path):
            try:
                # Keras 3 approach: TFSMLayer wraps SavedModel as inference layer
                m = tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
                return m, "SavedModel (TFSMLayer)"
            except Exception as e2:
                logger.warning(f"  {name} TFSMLayer failed: {e2}")
                try:
                    # Low-level tf.saved_model.load
                    m = tf.saved_model.load(savedmodel_path)
                    return m, "SavedModel (tf.saved_model)"
                except Exception as e3:
                    logger.warning(f"  {name} tf.saved_model failed: {e3}")

        raise FileNotFoundError(f"No loadable format found for {name}")

    # ── 1. Classifier ────────────────────────────────────────────────────────
    cls_dir        = os.path.join(MODELS_DIR, "classifier")
    cls_keras_path = os.path.join(cls_dir, "classifier.keras")
    cls_saved_path = os.path.join(cls_dir, "classifier_saved")
    tfidf_path     = os.path.join(cls_dir, "tfidf_vectorizer.joblib")
    nlp_path       = os.path.join(cls_dir, "nlp_preprocessor.joblib")

    if all(os.path.exists(p) for p in [tfidf_path, nlp_path]):
        try:
            # Classifier has multi-input (text + image) — use tf.saved_model.load()
            # instead of TFSMLayer which has issues with multi-input kwargs
            if os.path.exists(cls_keras_path):
                try:
                    cls_model = tf.keras.models.load_model(
                        cls_keras_path, custom_objects=CUSTOM_OBJECTS
                    )
                    method = ".keras"
                except Exception as e1:
                    logger.warning(f"  Classifier .keras failed: {e1}")
                    cls_model = None
            else:
                cls_model = None

            if cls_model is None and os.path.exists(cls_saved_path):
                sm = tf.saved_model.load(cls_saved_path)
                # Store the signature function — it takes text_input + image_input kwargs
                models["classifier_infer"] = sm.signatures['serving_default']
                cls_model = sm  # Store raw saved_model
                method = "SavedModel (signature)"

            if cls_model is None:
                raise FileNotFoundError("No loadable classifier found")

            tfidf = joblib.load(tfidf_path)
            nlp   = joblib.load(nlp_path)
            models["classifier"] = cls_model
            models["tfidf"]      = tfidf
            models["nlp"]        = nlp
            n_features = len(tfidf.vocabulary_)
            logger.info(f"  Classifier loaded via {method} (TF-IDF features: {n_features})")
        except Exception as e:
            logger.error(f"  Classifier gagal di-load: {e}")
    else:
        missing = [p for p in [tfidf_path, nlp_path] if not os.path.exists(p)]
        logger.warning(f"  Classifier artifacts tidak ditemukan: {missing}")

    # ── 2. Forecaster ────────────────────────────────────────────────────────
    fc_dir         = os.path.join(MODELS_DIR, "forecaster")
    fc_keras_path  = os.path.join(fc_dir, "forecaster.keras")
    fc_saved_path  = os.path.join(fc_dir, "forecaster_saved")
    fc_weights     = os.path.join(fc_dir, "forecaster_weights.h5")
    scaler_path    = os.path.join(fc_dir, "scaler.joblib")

    if os.path.exists(scaler_path):
        try:
            try:
                fc_model, method = _try_load_keras_or_savedmodel(
                    fc_keras_path, fc_saved_path, "Forecaster", CUSTOM_OBJECTS
                )
            except FileNotFoundError:
                # Last resort: rebuild architecture + load weights
                fc_model = build_forecaster()
                fc_model.load_weights(fc_weights)
                method = ".h5 weights (rebuild)"

            scaler = joblib.load(scaler_path)
            models["forecaster"] = fc_model
            models["scaler"]     = scaler
            logger.info(f"  Forecaster loaded via {method} (scaler n_features={scaler.n_features_in_})")
        except Exception as e:
            logger.error(f"  Forecaster gagal di-load: {e}")
    else:
        logger.warning("  Forecaster scaler tidak ditemukan.")

    # ── 3. OCR ───────────────────────────────────────────────────────────────
    ocr_dir        = os.path.join(MODELS_DIR, "ocr")
    ocr_keras_path = os.path.join(ocr_dir, "ocr_fixed.keras")
    ocr_h5_path    = os.path.join(ocr_dir, "ocr_weights_fixed.h5")

    try:
        # Try .keras first
        try:
            ocr_model = tf.keras.models.load_model(
                ocr_keras_path, custom_objects=CUSTOM_OBJECTS
            )
            method = ".keras"
        except Exception:
            # Rebuild CRNN architecture and load .h5 weights
            if os.path.exists(ocr_h5_path):
                ocr_model = _build_ocr_model()
                ocr_model.load_weights(ocr_h5_path)
                method = ".h5 weights (rebuild)"
            else:
                raise FileNotFoundError(f"No OCR model found at {ocr_keras_path} or {ocr_h5_path}")
        models["ocr"] = ocr_model
        logger.info(f"  OCR loaded via {method} (CRNN + CTC)")
    except Exception as e:
        logger.error(f"  OCR gagal di-load: {e}")

    # ── 4. Receipt Detector ──────────────────────────────────────────────────
    detector_keras = os.path.join(MODELS_DIR, "receipt_detector_best.keras")
    detector_saved = os.path.join(MODELS_DIR, "receipt_detector_savedmodel")

    try:
        det_model, method = _try_load_keras_or_savedmodel(
            detector_keras, detector_saved, "Receipt Detector", CUSTOM_OBJECTS
        )
        models["receipt_detector"] = det_model
        logger.info(f"  Receipt Detector loaded via {method}")
    except Exception as e:
        logger.error(f"  Receipt Detector gagal di-load: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info(f"Models loaded: {list(models.keys())}")
    logger.info(f"Categories: {CATEGORIES} ({NUM_CLASSES} total)")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    models.clear()
    logger.info("Models unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Spendly AI API",
    description=(
        "REST API untuk sistem manajemen keuangan Spendly AI.\n\n"
        "**4 Model ML:**\n"
        "- Receipt Detector: deteksi area struk dalam foto\n"
        "- OCR CRNN: ekstraksi teks dari foto struk\n"
        "- Multimodal Classifier (TF-IDF + CNN + SE): klasifikasi 9 kategori pengeluaran\n"
        "- LSTM + Attention Forecaster: prediksi pengeluaran mingguan\n\n"
        "**Fitur Tambahan:**\n"
        "- Gemini AI Insight: saran keuangan personal\n\n"
        "**Coding Camp 2026 powered by DBS Foundation — CC26-PSU276**"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "INDOMARET Total Rp 45.000 Sabun Mandi Pasta Gigi"}
        }


class ForecastRequest(BaseModel):
    history: List[List[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "history": [
                    [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000]
                ] * 12
            }
        }


class InsightRequest(BaseModel):
    spending_summary: dict

    class Config:
        json_schema_extra = {
            "example": {
                "spending_summary": {
                    "F&B": 500000,
                    "Groceries": 800000,
                    "Beauty": 150000,
                    "Transport": 120000,
                    "total": 1570000,
                }
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Status"])
def health_check():
    """Cek status server dan model yang ter-load."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "categories": CATEGORIES,
        "models_loaded": {
            "receipt_detector": "receipt_detector" in models,
            "ocr":              is_easyocr_available(),
            "crnn_ocr":         "ocr" in models,
            "classifier":       "classifier" in models,
            "forecaster":       "forecaster" in models,
        },
        "tensorflow": {
            "available": TF_AVAILABLE,
            "error": TF_IMPORT_ERROR,
        },
        "ocr_engine": {
            "primary": "easyocr",
            "languages": EASYOCR_LANGS,
            "gpu": EASYOCR_GPU,
            "reader_loaded": easyocr_reader is not None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /detect-receipt
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/detect-receipt", tags=["Receipt Detector"])
async def detect_receipt_endpoint(file: UploadFile = File(...)):
    """
    Deteksi area struk (bounding box) dalam foto menggunakan CNN.

    - **Input**: file gambar (JPG/PNG)
    - **Output**: bounding boxes, confidence scores, dan dimensi gambar asli
    """
    if "receipt_detector" not in models:
        raise HTTPException(
            status_code=503,
            detail="Receipt Detector model belum ter-load.",
        )

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detector = models["receipt_detector"]
    orig_h, orig_w = img_bgr.shape[:2]

    # Ambil input shape dari model secara dinamis
    target_h, target_w = get_model_input_size(detector)

    img_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
    predictions = model_predict(detector, img_tensor)

    # Format output tergantung arsitektur model
    # Asumsi output: bounding box coordinates (normalized) dan/atau confidence
    result = {
        "original_size": {"width": orig_w, "height": orig_h},
        "model_input_size": {"width": target_w, "height": target_h},
    }

    if isinstance(predictions, list):
        # Multi-output model
        result["raw_predictions"] = [p.tolist() for p in predictions]
    else:
        pred_array = predictions[0]  # batch dim
        if pred_array.ndim == 1:
            if len(pred_array) >= 4:
                # Output = [x_min, y_min, x_max, y_max, ...confidence]
                bbox = pred_array[:4].tolist()
                confidence = float(pred_array[4]) if len(pred_array) > 4 else 1.0
                # Denormalize bbox ke koordinat piksel asli
                result["detections"] = [
                    {
                        "bbox_normalized": {
                            "x_min": round(bbox[0], 4),
                            "y_min": round(bbox[1], 4),
                            "x_max": round(bbox[2], 4),
                            "y_max": round(bbox[3], 4),
                        },
                        "bbox_pixels": {
                            "x_min": int(bbox[0] * orig_w),
                            "y_min": int(bbox[1] * orig_h),
                            "x_max": int(bbox[2] * orig_w),
                            "y_max": int(bbox[3] * orig_h),
                        },
                        "confidence": round(confidence, 4),
                    }
                ]
            else:
                # Binary classification (receipt vs no receipt)
                confidence = float(pred_array[0]) if len(pred_array) == 1 else float(np.max(pred_array))
                result["is_receipt"] = bool(confidence > 0.5)
                result["confidence"] = round(confidence, 4)
        else:
            # Multiple detections or grid output
            result["raw_predictions"] = pred_array.tolist()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /ocr
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr", tags=["OCR"])
async def ocr_endpoint(
    file: Optional[UploadFile] = File(None),
    receipt: Optional[UploadFile] = File(None),
):
    """
    Ekstraksi teks dari gambar struk belanja menggunakan EasyOCR full receipt.

    - **Input**: file gambar (JPG/PNG)
    - **Output**: teks hasil OCR
    """
    upload = get_upload_file(file, receipt)
    contents = await upload.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ocr_result = run_easyocr(img_bgr)
    blur_info = assess_blur(img_bgr)

    return {
        "extracted_text": ocr_result["raw_text"],
        "raw_text":       ocr_result["raw_text"],
        "ocr_lines":      ocr_result["lines"],
        "ocr_engine":     "easyocr",
        "ocr_confidence": round(ocr_result["avg_confidence"], 4),
        "blur_score":     round(blur_info["blur_score"], 2),
        "is_blurry":      blur_info["is_blurry"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /classify
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/classify", tags=["Classifier"])
async def classify_endpoint(
    text: str = Form(default="", description="Teks OCR dari struk (bisa kosong jika gambar blurry)"),
    file: Optional[UploadFile] = File(None),
    receipt: Optional[UploadFile] = File(None),
):
    """
    Klasifikasi kategori pengeluaran dari teks struk + gambar.

    - **Input**: `text` (form field, teks OCR) + `file` (gambar struk)
    - **Output**: kategori (9 kelas), confidence score, semua probabilitas

    Gunakan `multipart/form-data`. Contoh curl:
    ```
    curl -X POST /classify -F "text=INDOMARET Total Rp 45000" -F "file=@struk.jpg"
    ```
    """
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Classifier model belum ter-load.")

    upload = get_upload_file(file, receipt)
    contents = await upload.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    blur_info  = assess_blur(img_bgr)
    classification = classify_spending(img_bgr, text, blur_info["is_blurry"])

    return {
        "category":   classification["category"],
        "confidence": classification["confidence"],
        "all_scores": classification["all_scores"],
        "is_blurry":  blur_info["is_blurry"],
        "blur_score": round(blur_info["blur_score"], 2),
        "text_used":  classification["text_used"] if not blur_info["is_blurry"] else "(blurry - CNN only)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /forecast
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/forecast", tags=["Forecaster"])
def forecast_endpoint(request: ForecastRequest):
    """
    Prediksi pengeluaran minggu depan berdasarkan histori 12 minggu.

    - **Input**: `history` — array 12×9 (12 minggu × 9 kategori, dalam Rupiah)
    - **Output**: prediksi pengeluaran minggu ke-13 per kategori (9 kategori)

    Urutan 9 kategori input: Beauty, F&B, Gas, Groceries, Health, HouseHold, Lifestyle, Listrik, Transport.
    """
    if "forecaster" not in models:
        raise HTTPException(status_code=503, detail="Forecaster model belum ter-load.")

    history = request.history
    if len(history) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"History harus berisi {LOOKBACK} minggu. Diterima: {len(history)}",
        )
    if any(len(row) != N_FORECAST_FEATURES for row in history):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Setiap minggu harus berisi {N_FORECAST_FEATURES} nilai "
                f"(satu per kategori: {CATEGORIES})."
            ),
        )

    scaler   = models["scaler"]
    fc_model = models["forecaster"]

    history_arr = np.array(history, dtype=np.float32)       # (12, 9)
    scaled      = scaler.transform(history_arr)              # normalize
    input_seq   = np.expand_dims(scaled, axis=0)             # (1, 12, 9)

    pred_scaled = model_predict(fc_model, input_seq)     # (1, 9)
    pred_real   = scaler.inverse_transform(pred_scaled)[0]   # (9,) Rupiah

    prediction = {}
    for i, cat in enumerate(CATEGORIES):
        prediction[cat] = round(float(pred_real[i]))

    return {
        "prediction": prediction,
        "unit":       "IDR (Rupiah)",
        "horizon":    "1 minggu ke depan",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /insight
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/insight", tags=["Generative AI"])
async def insight_endpoint(request: InsightRequest):
    """
    Saran keuangan personal menggunakan Google Gemini Generative AI.

    - **Input**: `spending_summary` — dict ringkasan pengeluaran per kategori
    - **Output**: saran keuangan berbahasa Indonesia

    *Membutuhkan GEMINI_API_KEY di environment variable.*
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY tidak ditemukan di environment.",
        )

    summary = request.spending_summary
    summary_str = "\n".join(
        f"  - {k}: Rp {v:,.0f}" if isinstance(v, (int, float)) else f"  - {k}: {v}"
        for k, v in summary.items()
    )

    prompt = (
        f"Kamu adalah asisten keuangan pribadi untuk generasi muda Indonesia.\n\n"
        f"Berikut ringkasan pengeluaran pengguna minggu ini:\n{summary_str}\n\n"
        f"Berikan 3 saran keuangan yang praktis, spesifik, dan mudah dipahami "
        f"berdasarkan data di atas. Gunakan bahasa Indonesia yang ramah dan motivatif. "
        f"Format: poin-poin singkat."
    )

    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            gemini_url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "maxOutputTokens": 512,
                    "temperature": 0.7,
                },
            },
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {response.text}",
        )

    result    = response.json()
    ai_advice = result["candidates"][0]["content"]["parts"][0]["text"]

    return {
        "spending_summary": summary,
        "insight":          ai_advice,
        "model_used":       "gemini-2.5-flash",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /process-receipt  (Full Pipeline: Detect → OCR → Classify)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/process-receipt", tags=["Full Pipeline"])
async def process_receipt_endpoint(
    file: Optional[UploadFile] = File(None),
    receipt: Optional[UploadFile] = File(None),
):
    """
    Full pipeline: Upload foto → Deteksi Struk → OCR → Klasifikasi Kategori.

    1. Receipt detection (CNN) — crop area struk jika terdeteksi
    2. OCR full receipt dengan EasyOCR
    3. Multimodal classification (teks OCR + gambar)
    4. Regex extraction untuk merchant, date, dan total amount

    - **Input**: file gambar struk (JPG/PNG)
    - **Output**: field scan yang kompatibel dengan frontend Spendly
    """
    upload = get_upload_file(file, receipt)
    contents = await upload.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Step 1: Receipt Detection (optional — jika model tersedia) ───────
    img_for_processing = img_bgr
    if "receipt_detector" in models:
        try:
            detector = models["receipt_detector"]
            target_h, target_w = get_model_input_size(detector)

            det_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
            det_preds  = model_predict(detector, det_tensor)

            # Coba crop bounding box jika output berupa koordinat
            if not isinstance(det_preds, list):
                pred_array = det_preds[0]
                if pred_array.ndim == 1 and len(pred_array) >= 4:
                    x_min = max(0, int(pred_array[0] * orig_w))
                    y_min = max(0, int(pred_array[1] * orig_h))
                    x_max = min(orig_w, int(pred_array[2] * orig_w))
                    y_max = min(orig_h, int(pred_array[3] * orig_h))

                    # Hanya crop jika bbox cukup besar (minimal 10% area asli)
                    bbox_area  = (x_max - x_min) * (y_max - y_min)
                    orig_area  = orig_w * orig_h
                    if bbox_area > 0.10 * orig_area and x_max > x_min and y_max > y_min:
                        img_for_processing = img_bgr[y_min:y_max, x_min:x_max]
                        logger.info(
                            f"Receipt cropped: ({x_min},{y_min}) → ({x_max},{y_max})"
                        )
        except Exception as e:
            logger.warning(f"Receipt detection gagal, pakai gambar asli: {e}")

    # ── Step 2: Full receipt OCR via EasyOCR ─────────────────────────────
    ocr_result = run_easyocr(img_for_processing)
    extracted_text = ocr_result["raw_text"]

    # ── Step 3: Multimodal classification ────────────────────────────────
    classification = classify_spending(img_for_processing, extracted_text, is_blurry=False)

    # ── Step 4: Regex extraction for frontend-ready fields ───────────────
    extracted_fields = extract_receipt_fields(ocr_result, classification)

    return {
        "status": "completed",
        **extracted_fields,
    }
