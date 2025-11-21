# api.py
import io
import base64
import os

import joblib
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from utils_audio import (
    normalize_audio,
    extract_features,
    embedding_from_mfcc,
    SR,
)

from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "military_voice_auth.joblib"

app = FastAPI(title="AI Voice Auth API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow frontend on localhost to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load artifacts once ------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

artifacts = joblib.load(MODEL_PATH)
spoof_clf = artifacts["spoof_clf"]
spoof_scaler = artifacts["spoof_scaler"]
iso = artifacts["iso_detector"]
enrolled_db = artifacts.get("enrolled_db", {})

# ------------------ Helper: make plot as base64 ------------------
def make_waveform_spectrogram(y, sr=SR):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))

    # waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")

    # mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64

# ------------------ Main analyze endpoint ------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()

    # decode audio
    try:
        y, sr = sf.read(io.BytesIO(raw))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception as e:
        return {"error": f"Could not decode audio: {e}"}

    # normalize + features
    y = normalize_audio(y)
    feat = extract_features(y)
    feat_s = spoof_scaler.transform(feat.reshape(1, -1))
    emb = embedding_from_mfcc(y)

    # spoof detector
    prob = spoof_clf.predict_proba(feat_s)[0]  # [prob_fake, prob_real]
    pred = int(spoof_clf.predict(feat_s)[0])
    prob_fake = float(prob[0])
    prob_real = float(prob[1]) if len(prob) > 1 else (1.0 - prob_fake)

    # anomaly score
    iso_score = None
    if iso is not None:
        try:
            iso_score = float(iso.decision_function(emb.reshape(1, -1))[0])
        except Exception:
            iso_score = None

    # speaker similarity (if any enrolled)
    best_id = None
    best_sim = 0.0
    if enrolled_db:
        for pid, rec in enrolled_db.items():
            ref = rec["embedding"].reshape(1, -1)
            sim = float(cosine_similarity(emb.reshape(1, -1), ref)[0, 0])
            if sim > best_sim:
                best_sim = sim
                best_id = pid

    # simple decision policy (frontend can still adjust)
    if pred == 1:
        verdict = "real"
    else:
        verdict = "fake"

    # coarse risk level from spoof probabilities
    if prob_fake >= 0.8:
        risk = "High"
    elif prob_fake >= 0.5:
        risk = "Medium"
    else:
        risk = "Low"

    # image
    plot_data_url = make_waveform_spectrogram(y, sr=SR)

    return {
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "pred": int(pred),
        "verdict": verdict,
        "risk_level": risk,
        "iso_score": iso_score,
        "best_match": best_id,
        "similarity": best_sim if enrolled_db else None,
        "has_enrollment": bool(enrolled_db),
        "plot_image": plot_data_url,
    }
