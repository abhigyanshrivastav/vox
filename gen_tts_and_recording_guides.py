import os
import json
import random
import asyncio
import edge_tts

# -----------------------------
# CONFIGURATION
# -----------------------------
VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural"
]

SENTENCES = [
    "Secure channels must remain uncompromised at all times.",
    "Authentication codes will rotate every five minutes.",
    "Drone surveillance detected movement near the perimeter.",
    "The quick brown fox jumps over the lazy dog.",
    "Mission authorization requires biometric verification.",
    "Radio silence must be maintained during infiltration.",
    "Encryption keys must never be transmitted verbally.",
    "All personnel must verify identity using voice prints."
]

DATASET_DIR = "dataset"
NUM_FAKE_SAMPLES = 10
NUM_REAL_SAMPLES = 10


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


async def generate_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


# -----------------------------
# GENERATE RECORDING GUIDE
# -----------------------------
def generate_recording_guide(speaker_id):
    guide_text = [
        f"Recording Guide for Speaker {speaker_id}",
        "--------------------------------------------",
        "Instructions:",
        "- Record in a quiet room.",
        "- Keep the mic 10–20 cm away.",
        "- Speak naturally, do not whisper.",
        "- Record each sentence clearly.",
        "",
        "Sentences to Record:"
    ]
    
    for idx, sentence in enumerate(SENTENCES, 1):
        guide_text.append(f"{idx}. {sentence}")

    guide_path = os.path.join(DATASET_DIR, f"speaker_{speaker_id}", "recording_guide.txt")
    with open(guide_path, "w") as f:
        f.write("\n".join(guide_text))


# -----------------------------
# GENERATE FAKE TTS SAMPLES
# -----------------------------
async def generate_fake_samples(speaker_id):
    fake_dir = os.path.join(DATASET_DIR, f"speaker_{speaker_id}", "fake")
    ensure_dir(fake_dir)

    for i in range(NUM_FAKE_SAMPLES):
        text = random.choice(SENTENCES)
        voice = random.choice(VOICES)
        file_path = os.path.join(fake_dir, f"fake_{i+1}.wav")
        await generate_tts(text, voice, file_path)


# -----------------------------
# GENERATE METADATA
# -----------------------------
def generate_metadata(speaker_id):
    meta = {
        "speaker_id": speaker_id,
        "num_real_samples_required": NUM_REAL_SAMPLES,
        "num_fake_samples": NUM_FAKE_SAMPLES,
        "voices_used": VOICES,
        "sentences": SENTENCES
    }

    meta_path = os.path.join(DATASET_DIR, f"speaker_{speaker_id}", "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)


# -----------------------------
# MAIN EXECUTION
# -----------------------------
async def main():
    speaker_id = input("Enter speaker ID (e.g., 01): ").strip()

    # Create folder structure
    base = os.path.join(DATASET_DIR, f"speaker_{speaker_id}")
    ensure_dir(base)
    ensure_dir(os.path.join(base, "real"))
    ensure_dir(os.path.join(base, "fake"))

    print("[+] Generating recording guide...")
    generate_recording_guide(speaker_id)

    print("[+] Generating TTS deepfake samples...")
    await generate_fake_samples(speaker_id)

    print("[+] Generating metadata...")
    generate_metadata(speaker_id)

    print("\nDONE.")
    print(f"Dataset ready under: {DATASET_DIR}/speaker_{speaker_id}")
    print("Record REAL samples manually using the recording_guide.txt file.")


if __name__ == "__main__":
    asyncio.run(main())
