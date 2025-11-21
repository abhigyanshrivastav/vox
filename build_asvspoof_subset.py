# build_asvspoof_subset.py
"""
Create a small, balanced subset of ASVspoof 2019 LA and copy it into:
    dataset/asv_train/{real,fake}
    dataset/asv_dev/{real,fake}

Adjust paths (ASV_ROOT and protocol paths) according to where you extracted the dataset.
"""

import os
import shutil

# ---- CONFIG: adjust these if your paths / filenames differ ----

# Root path where your ASVspoof2019_LA_* folders are
ASV_ROOT = "."  # current directory; change if needed

SUBSETS = [
    {
        "name": "train",
        "base_dir": "ASVspoof2019_LA_train",
        "flac_dir": "ASVspoof2019_LA_train/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "out_real": "dataset/asv_train/real",
        "out_fake": "dataset/asv_train/fake",
        "max_real": 100,
        "max_fake": 100,
    },
    {
        "name": "dev",
        "base_dir": "ASVspoof2019_LA_dev",
        "flac_dir": "ASVspoof2019_LA_dev/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "out_real": "dataset/asv_dev/real",
        "out_fake": "dataset/asv_dev/fake",
        "max_real": 100,
        "max_fake": 100,
    },
]

# If your protocol files are in a separate folder (e.g. ASVspoof2019_LA_cm_protocols),
# then change "protocol" above to something like:
# "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


def parse_protocol(path):
    """
    Parse ASVspoof protocol file.

    We try to be robust to slightly different formats.
    Typical lines look like:
        LA_T_1000137 - - bonafide
        LA_T_1000138 - - spoof A07
    or sometimes:
        speakerid utt_id system label [attack]

    Returns list of tuples: (utt_id, label_str)
    where label_str is "bonafide" or "spoof".
    """
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            # Find label token
            label = None
            for p in parts:
                if p.lower() == "bonafide":
                    label = "bonafide"
                    break
                if p.lower() == "spoof":
                    label = "spoof"
                    break
            if label is None:
                continue  # skip lines without clear bonafide/spoof

            # Guess utt_id column:
            # try first token, if no corresponding .flac exists,
            # try second token.
            utt_candidates = [parts[0]]
            if len(parts) > 1:
                utt_candidates.append(parts[1])

            out.append((utt_candidates, label))
    return out


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    for subset in SUBSETS:
        name = subset["name"]
        flac_dir = os.path.join(ASV_ROOT, subset["flac_dir"])
        protocol_path = os.path.join(ASV_ROOT, subset["protocol"])

        if not os.path.isdir(flac_dir):
            print(f"[ERROR] flac dir not found for {name}: {flac_dir}")
            continue
        if not os.path.isfile(protocol_path):
            print(f"[ERROR] protocol file not found for {name}: {protocol_path}")
            continue

        entries = parse_protocol(protocol_path)
        print(f"[INFO] Parsed {len(entries)} entries from protocol for {name}")

        out_real = subset["out_real"]
        out_fake = subset["out_fake"]
        ensure_dir(out_real)
        ensure_dir(out_fake)

        real_count = 0
        fake_count = 0

        for utt_candidates, lab in entries:
            # find actual file that exists
            flac_path = None
            for utt_id in utt_candidates:
                candidate = os.path.join(flac_dir, utt_id + ".flac")
                if os.path.isfile(candidate):
                    flac_path = candidate
                    break

            if flac_path is None:
                # Could not find a matching .flac for this protocol line
                # print(f"[WARN] no file found for IDs: {utt_candidates}")
                continue

            if lab == "bonafide":
                if real_count >= subset["max_real"]:
                    continue
                dest_dir = out_real
                real_count += 1
            else:  # spoof
                if fake_count >= subset["max_fake"]:
                    continue
                dest_dir = out_fake
                fake_count += 1

            dest_path = os.path.join(dest_dir, os.path.basename(flac_path))
            shutil.copy2(flac_path, dest_path)

            if real_count >= subset["max_real"] and fake_count >= subset["max_fake"]:
                break

        print(
            f"[DONE] {name}: copied {real_count} bonafide -> {out_real}, "
            f"{fake_count} spoof -> {out_fake}"
        )


if __name__ == "__main__":
    main()