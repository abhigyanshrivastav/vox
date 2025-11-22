# gen_fake_tts.py
#
# Generate fake (TTS) clips for your dataset from a list of sentences.
# It automatically continues numbering after the last existing fake_XX.wav.

import re
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment

TARGET_SR = 16000  # must match SR in utils_audio.py

SENTENCE_FILE = "sentences_fake.txt"
OUT_DIR = Path("dataset") / "speaker_01" / "fake"


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUT_DIR.resolve()}")


def load_sentences(path: str):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    print(f"[INFO] Loaded {len(sentences)} sentences from {path}")
    return sentences


def get_next_index():
    """
    Scan OUT_DIR for files named like fake_XX.wav or fake_XXX.wav
    and return the next integer index.
    """
    if not OUT_DIR.exists():
        return 1

    pattern = re.compile(r"fake_(\d+)\.wav$", re.IGNORECASE)
    max_idx = 0

    for p in OUT_DIR.glob("fake_*.wav"):
        m = pattern.search(p.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx

    next_idx = max_idx + 1
    print(f"[INFO] Highest existing fake index: {max_idx} -> next will be {next_idx}")
    return next_idx


def tts_to_wav(text: str, out_path: Path, lang="en", slow=False):
    # 1) Synthesize to temporary MP3 via gTTS
    tmp_mp3 = out_path.with_suffix(".mp3")
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(str(tmp_mp3))

    # 2) Convert MP3 -> WAV 16k mono
    audio = AudioSegment.from_mp3(tmp_mp3)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    audio.export(out_path, format="wav")
    tmp_mp3.unlink(missing_ok=True)


def main():
    ensure_out_dir()
    sentences = load_sentences(SENTENCE_FILE)
    if not sentences:
        print("[ERROR] No sentences found, aborting.")
        return

    start_idx = get_next_index()

    for offset, sent in enumerate(sentences):
        idx = start_idx + offset
        # keep at least 2 digits; if you prefer 3 digits, use {idx:03d}
        fname = f"fake_{idx:02d}.wav"
        out_path = OUT_DIR / fname
        print(f"[GEN] {fname}  ←  \"{sent}\"")

        try:
            tts_to_wav(sent, out_path)
        except Exception as e:
            print(f"[WARN] Failed to synthesize '{sent}': {e}")

    print("[DONE] Fake TTS generation complete.")


if __name__ == "__main__":
    main()
