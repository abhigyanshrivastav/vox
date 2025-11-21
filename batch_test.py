# batch_test.py
import os, joblib, numpy as np, hashlib
from utils_audio import load_audio, normalize_audio, extract_features, embedding_from_mfcc, SR
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "military_voice_auth.joblib"
DATASET = "dataset"

data = joblib.load(MODEL)
clf = data["spoof_clf"]
scaler = data["spoof_scaler"]
iso = data["iso_detector"]
enrolled = data.get("enrolled_db", {})

def classify_file(path):
    y = load_audio(path)
    y = normalize_audio(y)
    feat = extract_features(y).reshape(1,-1)
    feat_s = scaler.transform(feat)
    prob = clf.predict_proba(feat_s)[0]  # [prob_fake, prob_real]
    pred = clf.predict(feat_s)[0]
    emb = embedding_from_mfcc(y).reshape(1,-1)
    iso_score = None
    if iso is not None:
        try:
            iso_score = iso.decision_function(emb)[0]
        except:
            iso_score = None
    best_match = ("<none>", 0.0)
    for pid, rec in enrolled.items():
        sim = cosine_similarity(emb, rec["embedding"].reshape(1,-1))[0,0]
        if sim > best_match[1]:
            best_match = (pid, sim)
    return pred, prob, iso_score, best_match

for speaker in sorted(os.listdir(DATASET)):
    spath = os.path.join(DATASET, speaker)
    if not os.path.isdir(spath): continue
    for folder in ("real", "fake"):
        d = os.path.join(spath, folder)
        if not os.path.isdir(d): continue
        print("====", speaker, folder, "====")
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.wav','.mp3','.flac')): continue
            p = os.path.join(d, f)
            try:
                pred, prob, iso_score, best = classify_file(p)
                print(f"{f:25s} -> pred={pred} prob_real={prob[1]:.3f} prob_fake={prob[0]:.3f} iso={iso_score} best_match={best}")
            except Exception as e:
                print("ERR", f, e)
