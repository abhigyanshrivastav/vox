# app_military_voice_auth.py
import io
import os
import hashlib

import joblib
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
import librosa
import librosa.display

from sklearn.metrics.pairwise import cosine_similarity

from utils_audio import (
    load_audio,        # kept for consistency, even if not used directly
    normalize_audio,
    extract_features,
    embedding_from_mfcc,
    SR,
)

# --------------------------
# Streamlit basic config
# --------------------------
st.set_page_config(
    page_title="AI Voice Authentication & Anti-Deepfake Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# Session state init
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "challenge_phrase" not in st.session_state:
    st.session_state.challenge_phrase = None

if "example_bytes" not in st.session_state:
    st.session_state.example_bytes = None
    st.session_state.example_name = None

# --------------------------
# Load model artifacts
# --------------------------
MODEL_PATH = "military_voice_auth.joblib"

@st.cache_resource
def load_artifacts(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

artifacts = load_artifacts(MODEL_PATH)
spoof_clf = artifacts["spoof_clf"]
spoof_scaler = artifacts["spoof_scaler"]
iso = artifacts["iso_detector"]
enrolled_db = artifacts.get("enrolled_db", {})

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("🔧 Controls")

jury_mode = st.sidebar.checkbox(
    "Jury demo mode (simplified view)",
    value=True,
    help="Hide low-level debug info by default when presenting to jury."
)

st.sidebar.markdown("**Decision thresholds**")

similarity_threshold = st.sidebar.slider(
    "Speaker similarity (cosine)",
    0.60, 0.99, 0.80, 0.01,
    help="Higher = stricter identity match (fewer impostors, more false rejections)."
)

anomaly_threshold = st.sidebar.slider(
    "Anomaly score (IsolationForest)",
    -1.5, 1.5, 0.0, 0.05,
    help="Lower threshold = detector flags more samples as anomalous."
)

st.sidebar.markdown("---")
challenge_mode = st.sidebar.checkbox(
    "Enable challenge–response concept",
    value=True,
    help="Simulate secure challenge phrases to be spoken by the caller."
)

st.sidebar.markdown("**Demo examples**")
example_choice = st.sidebar.selectbox(
    "Load example sample",
    ["None", "Real: enrolled-like", "Fake: TTS-like"],
    help="Uses local example files if you place them under ./examples/."
)
if st.sidebar.button("Load selected example"):
    path = None
    if example_choice == "Real: enrolled-like":
        path = "examples/real_enrolled.wav"
    elif example_choice == "Fake: TTS-like":
        path = "examples/fake_tts.wav"

    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.session_state.example_bytes = f.read()
            st.session_state.example_name = os.path.basename(path)
        st.sidebar.success(f"Loaded example: {st.session_state.example_name}")
    elif example_choice != "None":
        st.sidebar.error("Example file not found. Put it under ./examples/.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
st.sidebar.write(f"- Enrolled identities: `{len(enrolled_db)}`")
if len(enrolled_db):
    st.sidebar.write("  - " + ", ".join(enrolled_db.keys()))
else:
    st.sidebar.caption("No explicit enrollment data; similarity is demo-only.")

# --------------------------
# Title / introduction
# --------------------------
st.markdown("## 🎤 AI-Based Voice Authentication & Anti-Deepfake Prototype")

st.markdown(
    """
> Scenario: **Authenticating critical voice commands** in a secure communication system (prototype).

This demo implements a **multi-layer defense** against deepfake / spoofed audio:

1. **Spoof detector** – ML model classifies human vs synthetic/spoofed speech  
2. **Anomaly detector** – checks if the signal embedding looks unusual vs known real voices  
3. **Speaker matching** – compares against enrolled voiceprint(s) using cosine similarity  
4. **Decision logic** – combines all to output an **ACCEPT / REJECT / SUSPICIOUS** verdict
"""
)

st.markdown("---")

# --------------------------
# Main layout with tabs
# --------------------------
tab_simple, tab_expert = st.tabs(["⚡ Quick Verdict", "🧠 Expert Analysis"])

# =========================================================
# QUICK VERDICT TAB
# =========================================================
with tab_simple:
    left, right = st.columns([1.2, 1])

    # ---------- LEFT: input section ----------
    with left:
        st.markdown("### 1️⃣ Provide audio: upload / record / example")

        # A. Upload file
        uploaded = st.file_uploader(
            "Upload an audio file (WAV / MP3 / FLAC / OGG / M4A)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Use a real recording or a TTS/deepfake clip."
        )

        # B. (Disabled) microphone recording – kept for future work
        st.markdown("##### 🎙️ Microphone recording (disabled in this demo)")
        st.caption("For the demo, please use uploaded audio files or examples.")

        mic_audio = None  # hard-disable mic path

        audio_bytes = None
        audio_name = None

        # Priority order:
        # 1) uploaded file
        # 2) loaded example
        if uploaded is not None:
            audio_bytes = uploaded.read()
            audio_name = uploaded.name
        elif st.session_state.example_bytes is not None:
            audio_bytes = st.session_state.example_bytes
            audio_name = st.session_state.example_name


        # Challenge–response section
        if challenge_mode:
            st.markdown("### 2️⃣ Challenge–Response (concept)")
            cols_ch = st.columns([2, 1])
            with cols_ch[0]:
                if st.button("Generate random phrase"):
                    import random
                    phrases = [
                        "Hold position at checkpoint",
                        "Authentication required for override",
                        "Alpha team proceed to sector three",
                        "Secure channel only",
                        "Abort mission on my signal",
                        "Confirm identity and continue",
                    ]
                    st.session_state.challenge_phrase = random.choice(phrases)
            with cols_ch[1]:
                if st.session_state.challenge_phrase:
                    st.success("Active challenge phrase:")
                    st.code(st.session_state.challenge_phrase)
                else:
                    st.caption(
                        "Click **Generate random phrase** to simulate a secure challenge "
                        "(in a real system, phonetic content would be verified too)."
                    )

    # ---------- RIGHT: verdict ----------
    with right:
        st.markdown("### 3️⃣ Verdict & Risk Overview")

        if audio_bytes is None:
            st.info("Upload, record, or load an example on the left to see analysis here.")
        else:
            # Core analysis
            audio_buf = io.BytesIO(audio_bytes)
            try:
                y, sr = sf.read(audio_buf)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
            except Exception as e:
                st.error(f"Could not read audio: {e}")
                y = None

            if y is not None:
                st.audio(audio_bytes)

                # preprocess
                y = normalize_audio(y)
                feat = extract_features(y)
                feat_s = spoof_scaler.transform(feat.reshape(1, -1))
                emb = embedding_from_mfcc(y)

                # spoof detector
                prob = spoof_clf.predict_proba(feat_s)[0]  # [prob_fake, prob_real]
                pred = int(spoof_clf.predict(feat_s)[0])
                prob_fake = float(prob[0])
                prob_real = float(prob[1]) if len(prob) > 1 else (1 - prob_fake)

                # anomaly detector
                iso_score = None
                if iso is not None:
                    try:
                        iso_score = float(iso.decision_function(emb.reshape(1, -1))[0])
                    except Exception:
                        iso_score = None

                # speaker similarity
                best_id = None
                best_sim = 0.0
                if len(enrolled_db):
                    for pid, rec in enrolled_db.items():
                        ref = rec["embedding"].reshape(1, -1)
                        sim = float(cosine_similarity(emb.reshape(1, -1), ref)[0, 0])
                        if sim > best_sim:
                            best_sim = sim
                            best_id = pid

                # decision logic
                reasons = []
                if prob_real < 0.6:
                    reasons.append("Spoof model: low real probability")
                if iso_score is not None and iso_score < anomaly_threshold:
                    reasons.append("Anomaly detector: unusual embedding")
                if len(enrolled_db) and best_sim < similarity_threshold:
                    reasons.append("Speaker mismatch: low similarity")

                # base accept/reject from spoof classifier
                if pred == 0 and prob_fake > 0.6:
                    base_verdict = "REJECT"
                else:
                    base_verdict = "ACCEPT"

                # final verdict category
                if base_verdict == "ACCEPT" and len(reasons) == 0:
                    verdict = "ACCEPT"
                    verdict_style = "✅ AUTHENTIC"
                elif base_verdict == "REJECT":
                    verdict = "REJECT"
                    verdict_style = "❌ SUSPECTED SPOOF"
                else:
                    verdict = "SUSPICIOUS"
                    verdict_style = "⚠️ SUSPICIOUS / REVIEW"

                # overall risk level
                risk = 0
                if prob_real < 0.6:
                    risk += 1
                if iso_score is not None and iso_score < anomaly_threshold:
                    risk += 1
                if len(enrolled_db) and best_sim < similarity_threshold:
                    risk += 1

                if risk == 0:
                    risk_level = "Low"
                elif risk == 1:
                    risk_level = "Medium"
                else:
                    risk_level = "High"

                # store in session history
                st.session_state.history.append({
                    "file": audio_name or "example",
                    "verdict": verdict,
                    "prob_real": round(prob_real, 3),
                    "prob_fake": round(prob_fake, 3),
                    "iso_score": round(iso_score, 3) if iso_score is not None else None,
                    "best_match": best_id,
                    "similarity": round(best_sim, 3) if len(enrolled_db) else None,
                    "risk_level": risk_level,
                })

                # show verdict prominently
                if verdict == "ACCEPT":
                    st.success(f"FINAL VERDICT: {verdict_style}")
                elif verdict == "REJECT":
                    st.error(f"FINAL VERDICT: {verdict_style}")
                else:
                    st.warning(f"FINAL VERDICT: {verdict_style}")

                # metrics cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Prob. REAL", f"{prob_real:.2f}")
                with c2:
                    st.metric("Prob. FAKE", f"{prob_fake:.2f}")
                with c3:
                    st.metric("Risk level", risk_level)

                if not jury_mode:
                    st.caption(f"Anomaly score (IsolationForest): {iso_score}")

                if len(enrolled_db):
                    st.write(
                        f"**Best match:** `{best_id}` "
                        f"with similarity **{best_sim:.3f}** "
                        f"(threshold {similarity_threshold:.2f})"
                    )
                else:
                    st.caption("No enrolled identities loaded – identity match not enforced.")

# =========================================================
# EXPERT TAB
# =========================================================
with tab_expert:
    st.markdown("### Detailed Analysis & Visualizations")

    # we reuse the last analyzed sample if available
    if not st.session_state.history:
        st.info("Analyze a sample in the **Quick Verdict** tab first.")
    else:
        last = st.session_state.history[-1]
        st.write(f"**Last sample analyzed:** {last['file']} (verdict: `{last['verdict']}`)")

        # For visuals, reconstruct from the last audio processed if still in scope
        # Note: For a fully robust app, you'd re-load the audio by path / bytes.
        # Here we reuse `y` from the last run in this session.
        if 'y' not in locals() or y is None:
            st.warning("Waveform not available in this session context.")
        else:
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown("#### Waveform & Spectrogram")
                fig, ax = plt.subplots(1, 2, figsize=(11, 3))

                # waveform
                librosa.display.waveshow(y, sr=SR, ax=ax[0])
                ax[0].set_title("Waveform")
                ax[0].set_xlabel("Time")
                ax[0].set_ylabel("Amplitude")

                # spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=SR)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, sr=SR, x_axis="time", y_axis="mel", ax=ax[1])
                ax[1].set_title("Mel Spectrogram (dB)")
                fig.colorbar(img, ax=ax[1], format="%+2.f dB")

                st.pyplot(fig)

            with col2:
                st.markdown("#### Internal Metrics (last sample)")
                st.write(f"- Prob. real: {last['prob_real']}")
                st.write(f"- Prob. fake: {last['prob_fake']}")
                st.write(f"- Anomaly score: {last['iso_score']}")
                st.write(f"- Risk level: {last['risk_level']}")
                st.write(f"- Best match: {last['best_match']}")
                st.write(f"- Similarity: {last['similarity']}")

                emb = embedding_from_mfcc(y)
                hash_hex = hashlib.sha256(emb.tobytes()).hexdigest()
                st.caption(f"Embedding hash (for integrity / logging): `{hash_hex[:16]}...`")

        with st.expander("ℹ️ How the decision is made (technical view)", expanded=not jury_mode):
            st.markdown(
                """
- **Features**: MFCC + spectral features extracted from the audio.
- **Spoof detector**: RandomForest classifier predicts real vs fake/spoof.
- **Anomaly detector**: IsolationForest trained only on real embeddings; low scores ⇒ unusual audio.
- **Speaker matcher**: cosine similarity between sample embedding and enrolled voiceprint(s).
- **Decision rule**: combines spoof probability, anomaly score, and similarity against thresholds to output ACCEPT / REJECT / SUSPICIOUS.
                """
            )

        with st.expander("🧪 Debug values (for development)", expanded=not jury_mode):
            if 'prob_real' in locals():
                st.write("Raw probabilities:", {"prob_fake": prob_fake, "prob_real": prob_real})
                st.write("Current thresholds:")
                st.write(f"- Similarity threshold: {similarity_threshold}")
                st.write(f"- Anomaly threshold: {anomaly_threshold}")
            else:
                st.caption("No debug info – analyze a clip first.")

# --------------------------
# Session History table
# --------------------------
st.markdown("---")
st.markdown("### 📜 Session History (this session)")

if len(st.session_state.history):
    st.dataframe(st.session_state.history)
else:
    st.caption("No samples analyzed yet. Upload, record, or load an example to start logging.")
