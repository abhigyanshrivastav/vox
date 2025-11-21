# test_load.py
import os
from utils_audio import load_audio

p = "dataset/speaker_01/real"
for f in sorted(os.listdir(p)):
    if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a')):
        path = os.path.join(p, f)
        print("Trying:", path)
        try:
            y = load_audio(path)
            print("  OK — samples:", len(y))
        except Exception as e:
            print("  FAILED:", repr(e))
