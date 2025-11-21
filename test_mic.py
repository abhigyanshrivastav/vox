import streamlit as st
import soundfile as sf
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

st.title("Mic test")

audio = st.audio_input("Record something")

if audio is not None:
    raw = audio.getvalue()
    st.write(f"Bytes captured: {len(raw)}")

    # play back raw audio
    st.audio(raw)

    # try to decode with soundfile
    try:
        y, sr = sf.read(io.BytesIO(raw))
        st.write(f"Decoded samples: {len(y)}, sr={sr}")
        if len(y) > 0:
            fig, ax = plt.subplots()
            ax.plot(y)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"soundfile failed: {e}")
