# convert_to_wav.py
import os
from pydub import AudioSegment

src = "dataset/speaker_01/real"
out = src  # overwrite into same folder
for fname in os.listdir(src):
    if not fname.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a')): 
        continue
    path = os.path.join(src, fname)
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        out_path = os.path.join(out, os.path.splitext(fname)[0] + ".wav")
        audio.export(out_path, format="wav")
        print("Converted:", out_path)
    except Exception as e:
        print("Failed convert", path, e)
