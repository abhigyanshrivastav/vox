# train_military_voice_auth.py
"""
Robust training pipeline for the military-grade voice authentication prototype.

Expect folder structure:
project/
  utils_audio.py
  train_military_voice_auth.py
  dataset/
    speaker_01/
      real/
      fake/
      recording_guide.txt
      metadata.json
  enroll/    # optional; subfolders per identity: enroll/<id>/*.wav

Run:
    python train_military_voice_auth.py
"""

import os
import sys
import numpy as np
import joblib
import hashlib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# utils_audio must exist in project
from utils_audio import load_audio, normalize_audio, extract_features, embedding_from_mfcc

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "dataset"      # expected to contain speaker subfolders
ENROLL_DIR = "enroll"     # optional - contains subfolders per enrolled identity
MODEL_OUT = "military_voice_auth.joblib"
MIN_SAMPLES_FOR_CV = 3    # fallback if dataset is tiny

# -----------------------
# Helper: audio file check
# -----------------------
def is_audio_file(name):
    name = name.lower()
    return name.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))

# -----------------------
# Load dataset: robust, nested structure
# -----------------------
def load_spoof_dataset(data_dir):
    X, y = [], []
    embeddings, labels_emb = [], []

    if not os.path.isdir(data_dir):
        raise SystemExit(f"[ERROR] Dataset directory not found: {data_dir}")

    num_real = 0
    num_fake = 0
    speakers_seen = 0

    for speaker in sorted(os.listdir(data_dir)):
        spk_path = os.path.join(data_dir, speaker)
        if not os.path.isdir(spk_path):
            continue
        speakers_seen += 1
        real_dir = os.path.join(spk_path, "real")
        fake_dir = os.path.join(spk_path, "fake")

        # load real samples
        if os.path.isdir(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if not is_audio_file(fname):
                    continue
                fpath = os.path.join(real_dir, fname)
                try:
                    audio = load_audio(fpath)
                except Exception as e:
                    print(f"[WARN] Failed to load real audio {fpath}: {e}")
                    continue
                audio = normalize_audio(audio)
                feat = extract_features(audio)
                emb = embedding_from_mfcc(audio)
                X.append(feat)
                y.append(1)
                embeddings.append(emb)
                labels_emb.append(1)
                num_real += 1

        # load fake samples
        if os.path.isdir(fake_dir):
            for fname in sorted(os.listdir(fake_dir)):
                if not is_audio_file(fname):
                    continue
                fpath = os.path.join(fake_dir, fname)
                try:
                    audio = load_audio(fpath)
                except Exception as e:
                    print(f"[WARN] Failed to load fake audio {fpath}: {e}")
                    continue
                audio = normalize_audio(audio)
                feat = extract_features(audio)
                emb = embedding_from_mfcc(audio)
                X.append(feat)
                y.append(0)
                embeddings.append(emb)
                labels_emb.append(0)
                num_fake += 1

    print(f"[INFO] Speakers discovered: {speakers_seen}; real samples: {num_real}; fake samples: {num_fake}")
    if len(X) == 0:
        raise SystemExit("[ERROR] No audio samples were loaded. Check dataset/ structure and audio files.")

    X = np.array(X)
    y = np.array(y).astype(int)
    embeddings = np.array(embeddings)

    return X, y, embeddings, np.array(labels_emb)

# -----------------------
# Enrollment loader
# -----------------------
def load_enrolled_db(enroll_dir):
    enrolled_db = {}
    if not os.path.isdir(enroll_dir):
        print(f"[WARN] Enrollment directory not found: {enroll_dir} — continuing without enrolled identities.")
        return enrolled_db

    for person in sorted(os.listdir(enroll_dir)):
        pdir = os.path.join(enroll_dir, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for fname in sorted(os.listdir(pdir)):
            if not is_audio_file(fname):
                continue
            path = os.path.join(pdir, fname)
            try:
                y_a = load_audio(path)
            except Exception as e:
                print(f"[WARN] Could not load enrollment audio {path}: {e}")
                continue
            y_a = normalize_audio(y_a)
            emb = embedding_from_mfcc(y_a)
            embs.append(emb)
        if len(embs) == 0:
            print(f"[WARN] No valid enrollment audio for identity {person}; skipping.")
            continue
        mean_emb = np.mean(np.vstack(embs), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        emb_bytes = mean_emb.tobytes()
        emb_hash = hashlib.sha256(emb_bytes).hexdigest()
        enrolled_db[person] = {"embedding": mean_emb, "hash": emb_hash, "n_samples": len(embs)}
    return enrolled_db

# -----------------------
# Train pipeline
# -----------------------
def train():
    print("[*] Loading dataset...")
    X, y, embeddings, labels_emb = load_spoof_dataset(DATA_DIR)

    # Basic stats
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[INFO] Label distribution: {label_counts}")

    # Scale features for spoof detector
    scaler = StandardScaler()
    try:
        X_s = scaler.fit_transform(X)
    except Exception as e:
        print(f"[ERROR] Failed to scale features: {e}")
        sys.exit(1)

    # Train/test split
    if len(y) < 2:
        raise SystemExit("[ERROR] Not enough labeled samples to train.")
    test_size = 0.2 if len(y) >= 5 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    # Spoof detector (Random Forest)
    print("[*] Training spoof detector (RandomForest)...")
    clf_spoof = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_spoof.fit(X_train, y_train)

    y_pred = clf_spoof.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Spoof detector accuracy on hold-out: {acc:.4f}")
    print("[RESULT] Classification report:\n", classification_report(y_test, y_pred))
    print("[RESULT] Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Cross validation (safe fallback if dataset too small)
    cv = 5
    if len(y) < MIN_SAMPLES_FOR_CV:
        cv = max(2, len(y))
    try:
        scores = cross_val_score(clf_spoof, X_s, y, cv=cv)
        print(f"[INFO] {cv}-fold CV spoof accuracy: {np.mean(scores):.4f}  {scores}")
    except Exception as e:
        print(f"[WARN] cross_val_score failed: {e}")

    # Train anomaly detector (IsolationForest) on real embeddings
    print("[*] Training anomaly detector on real embeddings (IsolationForest)...")
    real_embs = embeddings[labels_emb == 1]
    if len(real_embs) < 2:
        print("[WARN] Not enough real embeddings for IsolationForest; training skipped.")
        iso = None
    else:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        iso.fit(real_embs)

    # Load enrolled identities (if available)
    enrolled_db = load_enrolled_db(ENROLL_DIR)
    print(f"[INFO] Enrolled identities loaded: {list(enrolled_db.keys())}")

    # Save model artifacts
    artifacts = {
        "spoof_clf": clf_spoof,
        "spoof_scaler": scaler,
        "iso_detector": iso,
        "enrolled_db": enrolled_db
    }
    joblib.dump(artifacts, MODEL_OUT)
    print(f"[+] Saved artifacts to {MODEL_OUT}")

if __name__ == "__main__":
    train()
