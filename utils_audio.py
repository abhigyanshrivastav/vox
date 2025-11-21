# utils_audio.py
import numpy as np
import librosa
import soundfile as sf

SR = 22050
N_MFCC = 26

def load_audio(path, sr=SR):
    y, _sr = librosa.load(path, sr=sr)
    # mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # trim silence ends to reduce noise
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def normalize_audio(y):
    # peak normalize to 0.99
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.99
    return y

def extract_features(y, sr=SR, n_mfcc=N_MFCC):
    # core features used for spoof detector & basic embedding
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    feat = np.hstack([mfcc_mean, mfcc_std, zcr, spec_cent, spec_bw, rolloff, rms])
    return feat

def embedding_from_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    # lightweight embedding: mean + std of MFCC (works for demo)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    emb = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    # l2 normalize
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm
