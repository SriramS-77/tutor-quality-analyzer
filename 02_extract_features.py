"""
Step 2 - turn each chunk into a fixed numeric feature vector.

 TEXT  chunks -> TEXT_MODEL  (cheap Flash-Lite)  + deterministic transcript stats (no API)
 VIDEO chunks -> VIDEO_MODEL (multimodal Flash)  - sees video AND audio
The two modalities use different prompts/features and are extracted independently.
Every API response is cached in work/features/, so you can stop and re-run (free-tier quotas!).

  export GEMINI_API_KEY=...
  python 02_extract_features.py                 # both
  python 02_extract_features.py --only text
  python 02_extract_features.py --only video --video-rpm 4
"""
import argparse, json, os, re, time
import pandas as pd
from google import genai
from google.genai import types
from common import *
from feature_spec import *

TextModel = make_model("TextCoding", TEXT_FEATURES)
VideoModel = make_model("VideoCoding", VIDEO_FEATURES)
TECH = re.compile(r"can(?:not|'t)? you hear|can you see|on mute|unmute|frozen|freez|network|connection|"
                  r"internet|breaking up|not (?:audible|clear|visible)|come again|say that again|you are cut", re.I)
_last = {}


def throttle(key, rpm):
    gap = 60.0 / rpm - (time.time() - _last.get(key, 0))
    if gap > 0:
        time.sleep(gap)
    _last[key] = time.time()


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            m = str(e)
            if any(k in m for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE", "timed out")):
                w = min(90, 10 * 2 ** i)
                print(f"   retry in {w}s: {m[:90]}")
                time.sleep(w)
            else:
                raise
    raise RuntimeError("gave up after retries (daily quota? re-run later - cache is kept)")


# ------------------------------------------------------------------ text
def det_features(cues, start, end):
    dur = max(1.0, end - start)
    wc = lambda cs: sum(len(c[3].split()) for c in cs)
    tut = [c for c in cues if c[2] == "TUTOR"]
    stu = [c for c in cues if c[2] and c[2] != "TUTOR"]
    labelled = any(c[2] for c in cues)
    gaps = [b[0] - a[1] for a, b in zip(cues, cues[1:])]
    mono, cur, last_end = 0.0, 0.0, None                       # longest uninterrupted tutor run
    for c in cues:
        if c[2] == "TUTOR":
            cur = (cur + c[1] - c[0]) if (last_end is not None and c[0] - last_end < 3) else c[1] - c[0]
            last_end = c[1]; mono = max(mono, cur)
        else:
            cur, last_end = 0.0, None
    text = " ".join(c[3] for c in cues)
    tw, sw = wc(tut), wc(stu)
    return {
        "words_per_min": wc(cues) / dur * 60,
        "speech_coverage": min(1.0, sum(c[1] - c[0] for c in cues) / dur),
        "max_silence_s": max(gaps + [cues[0][0] - start if cues else dur]),
        "tutor_word_share": tw / (tw + sw) if labelled and tw + sw else float("nan"),
        "student_turns_per_min": len(stu) / dur * 60 if labelled else float("nan"),
        "mean_student_turn_words": sw / len(stu) if stu else 0.0,
        "longest_tutor_run_s": mono,
        "qmarks_per_min": text.count("?") / dur * 60,
        "tech_phrases": len(TECH.findall(text)),
    }


def text_prompt(ch):
    lines = [f"[{int(c[0])//60:02d}:{int(c[0])%60:02d}] {c[2] or 'SPEAKER'}: {c[3]}" for c in ch["cues"]]
    return (f"{RULES}\n\nThis is a transcript segment ({(ch['end']-ch['start'])/60:.1f} min) from an online "
            f"lesson. TUTOR is the teacher; STUDENT_n are learners (labels may be missing).\n\n" + "\n".join(lines))


def run_text(args, rec_ids, client):
    out = WORK / "features" / "text"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid in rec_ids:
        for f in sorted((WORK / "chunks" / "text" / rid).glob("*.json")):
            ch = json.load(open(f)); cache = out / f"{rid}_{ch['chunk']:03d}.json"
            llm_feats = None
            if cache.exists():
                llm_feats = json.load(open(cache))
            elif sum(len(c[3].split()) for c in ch["cues"]) < 20:
                llm_feats = dict(clean(TEXT_FEATURES, {}), evidence="(no speech)", empty=1)
                json.dump(llm_feats, open(cache, "w"))
            elif client:
                print(f"[text] {rid} chunk {ch['chunk']}")
                def call():
                    throttle("text", args.text_rpm)
                    r = client.models.generate_content(model=TEXT_MODEL, contents=text_prompt(ch),
                        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                           response_schema=TextModel))
                    return json.loads(r.text)
                d = with_retry(call)
                llm_feats = dict(clean(TEXT_FEATURES, d), evidence=d.get("evidence", ""), empty=0)
                json.dump(llm_feats, open(cache, "w"))
            if llm_feats is None:
                continue
            row = {"rec_id": rid, "chunk": ch["chunk"], "pos": ch["pos"], "t_empty": llm_feats.get("empty", 0)}
            row.update({f"t_{k}": v for k, v in det_features(ch["cues"], ch["start"], ch["end"]).items()})
            row.update({f"t_{n}": llm_feats[n] for n, _, _ in TEXT_FEATURES})
            row["t_evidence"] = llm_feats.get("evidence", "")
            rows.append(row)
    pd.DataFrame(rows).to_csv(WORK / "features_text.csv", index=False)
    print(f"wrote features_text.csv ({len(rows)} chunks)")


# ------------------------------------------------------------------ video
def run_video(args, rec_ids, client):
    out = WORK / "features" / "video"; out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(WORK / "manifest.csv").set_index("rec_id")
    rows = []
    for rid in rec_ids:
        last = max(1, int(man.loc[rid, "last_chunk"]))
        for f in sorted((WORK / "chunks" / "video" / rid).glob("*.mp4")):
            idx = int(f.stem); cache = out / f"{rid}_{idx:03d}.json"
            if not cache.exists():
                if not client:
                    continue
                print(f"[video] {rid} chunk {idx}")
                def call():
                    up = client.files.upload(file=str(f))
                    try:
                        while up.state.name == "PROCESSING":
                            time.sleep(3); up = client.files.get(name=up.name)
                        if up.state.name != "ACTIVE":
                            raise RuntimeError(f"upload state {up.state.name}")
                        throttle("video", args.video_rpm)
                        r = client.models.generate_content(model=VIDEO_MODEL,
                            contents=[up, RULES + "\n\nThis is a video segment of an online lesson for children. "
                                      "Code the video and its audio track."],
                            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                response_schema=VideoModel, media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW))
                        return json.loads(r.text)
                    finally:
                        try: client.files.delete(name=up.name)
                        except Exception: pass
                d = with_retry(call)
                json.dump(dict(clean(VIDEO_FEATURES, d), evidence=d.get("evidence", "")), open(cache, "w"))
            d = json.load(open(cache))
            row = {"rec_id": rid, "chunk": idx, "pos": idx / last}
            row.update({f"v_{n}": d[n] for n, _, _ in VIDEO_FEATURES})
            row["v_evidence"] = d.get("evidence", "")
            rows.append(row)
    pd.DataFrame(rows).to_csv(WORK / "features_video.csv", index=False)
    print(f"wrote features_video.csv ({len(rows)} chunks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["text", "video", "all"], default="all")
    ap.add_argument("--text-rpm", type=float, default=10)    # free tier Flash-Lite is ~15 RPM; keep margin
    ap.add_argument("--video-rpm", type=float, default=5)    # video chunks are token heavy
    ap.add_argument("--no-api", action="store_true", help="only rebuild CSVs from the cache")
    a = ap.parse_args()
    client = None if a.no_api else genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ids = list(pd.read_csv(WORK / "manifest.csv")["rec_id"])
    print(f"text model: {TEXT_MODEL} | video model: {VIDEO_MODEL}")
    if a.only in ("text", "all"): run_text(a, ids, client)
    if a.only in ("video", "all"): run_video(a, ids, client)
