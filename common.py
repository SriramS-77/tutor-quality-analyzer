"""Shared config + helpers. Everything is overridable with environment variables."""
import os, re, json
from pathlib import Path

# import shutil
# import sys
# for tool in ("ffmpeg", "ffprobe"):
#     if shutil.which(tool) is None:
#         sys.exit(f"'{tool}' not found on PATH. Install ffmpeg (winget install Gyan.FFmpeg), then open a NEW terminal.")

DATA_DIR = Path("./class recordings")
WORK = Path(os.environ.get("WORK_DIR", "work"))

CLASSES = ["weak", "average", "good", "excellent"]      # ordinal order, 0..3
ALIASES = {"poor": "weak"}
SKIP_DIRS = {"demo"}                                    # ignored completely
# Any other folder (e.g. "known_issues") is treated as UNLABELED: it is chunked and
# featurised, but never used for training/testing - only predicted at the end.

CHUNK_SEC = int(os.environ.get("CHUNK_SEC", 300))        # 5-minute chunks
MIN_CHUNK_SEC = 45                                       # keep short closing chunk (recap/homework live there)

# Check `python 00_list_models.py` for what your key can use and override these.
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-2.5-flash-lite")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "gemini-2.5-flash")


def discover():
    """Return [{rec_id,label,folder,mp4,vtt}] for every recording under DATA_DIR."""
    recs = []
    for d in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        name = d.name.lower().strip()
        if name in SKIP_DIRS:
            continue
        name = ALIASES.get(name, name)
        label = name if name in CLASSES else "unlabeled"
        tag = name if label != "unlabeled" else re.sub(r"\W+", "_", name)
        files = [f for f in sorted(d.iterdir()) if f.is_file()]
        mp4s = [f for f in files if f.suffix.lower() == ".mp4"]
        vtts = [f for f in files if f.suffix.lower() == ".vtt"]
        for i, mp4 in enumerate(mp4s, 1):
            vtt = next((v for v in vtts if v.stem == mp4.stem), None)
            if vtt is None and len(mp4s) == len(vtts):   # fall back to sorted-order pairing
                vtt = vtts[i - 1]
                print(f"[warn] pairing {mp4.name} <-> {vtt.name} by order")
            recs.append(dict(rec_id=f"{tag}_{i:02d}", label=label, folder=d.name,
                             mp4=str(mp4), vtt=str(vtt) if vtt else ""))
    return recs


# ---------------------------------------------------------------- VTT ----
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
_SPK = re.compile(r"^([^:]{1,40}):\s+(.*)$")
_VTAG = re.compile(r"<v\s+([^>]+)>")


def _sec(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(path):
    """-> list of (start, end, speaker_or_None, text)."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="ignore").replace("\r", "")
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for i, l in enumerate(lines):
            m = _TS.search(l)
            if m:
                g = m.groups()
                s, e = _sec(*g[0:4]), _sec(*g[4:8])
                body = " ".join(lines[i + 1:])
                spk = None
                v = _VTAG.search(body)
                if v:
                    spk = v.group(1).strip()
                body = re.sub(r"<[^>]+>", "", body).strip()
                if body:
                    cues.append([s, e, spk, body])
                break
    # "Name: text" style speaker labels (Zoom/Teams) - only trust if most cues have them
    if cues and sum(1 for c in cues if c[2]) == 0:
        hits = [_SPK.match(c[3]) for c in cues]
        if sum(1 for h in hits if h) >= 0.5 * len(cues):
            for c, h in zip(cues, hits):
                if h:
                    c[2], c[3] = h.group(1).strip(), h.group(2).strip()
    return cues


def anonymise(cues):
    """Replace real names: the speaker with most words = TUTOR (assumption), others STUDENT_k."""
    words = {}
    for _, _, spk, t in cues:
        if spk:
            words[spk] = words.get(spk, 0) + len(t.split())
    order = sorted(words, key=words.get, reverse=True)
    mp = {s: ("TUTOR" if i == 0 else f"STUDENT_{i}") for i, s in enumerate(order)}
    return [[s, e, mp.get(spk) if spk else None, t] for s, e, spk, t in cues]
