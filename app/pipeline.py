"""Server-side pipeline: cut chunks -> Gemini video features -> RF -> class, confidence, reasons."""
import json, re, shutil, subprocess, tempfile, time, math
from pathlib import Path
import numpy as np
from feature_spec import VIDEO_FEATURES, RULES, make_model, clean

NEG = {"v_tech_disruption", "v_background_noise", "v_tutor_audio_cutoffs"}          # higher = worse
NEUTRAL = {"pos", "v_screen_share_active", "v_learners_visible", "v_tutor_speaking_frac", "v_learner_speaking_frac"}
MIN_CHUNK_SEC = 45
_VideoModel = make_model("VideoCoding", VIDEO_FEATURES)
_DESC = {f"v_{n}": d for n, _, d in VIDEO_FEATURES}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


def duration(path):
    r = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise RuntimeError("could not read video duration - is this a valid mp4?")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def plan_chunks(dur, chunk_sec):
    n = math.ceil(dur / chunk_sec)
    keep = [i for i in range(n) if min((i + 1) * chunk_sec, dur) - i * chunk_sec >= MIN_CHUNK_SEC]
    return [(i, i * chunk_sec, min((i + 1) * chunk_sec, dur), i / max(1, keep[-1])) for i in keep]


def pick(chunks, max_chunks):
    """Evenly spaced subset (always keeps first and last) so quota/time stay bounded."""
    if max_chunks >= len(chunks):
        return chunks
    idx = np.unique(np.round(np.linspace(0, len(chunks) - 1, max_chunks)).astype(int))
    return [chunks[i] for i in idx]


def cut(src, start, end, out):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(end - start), "-i", src,
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=-2:360,fps=1", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "33", "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "16000", "-sn", "-dn", str(out)]
    subprocess.run(cmd, check=True)


def gemini_features(client, model, path, rpm=5, _last=[0.0]):
    from google.genai import types
    for attempt in range(6):
        up = None
        try:
            up = client.files.upload(file=str(path))
            while up.state.name == "PROCESSING":
                time.sleep(3); up = client.files.get(name=up.name)
            if up.state.name != "ACTIVE":
                raise RuntimeError(f"upload state {up.state.name}")
            wait = 60.0 / rpm - (time.time() - _last[0])
            if wait > 0: time.sleep(wait)
            _last[0] = time.time()
            r = client.models.generate_content(
                model=model,
                contents=[up, RULES + "\n\nThis is a video segment of an online lesson for children. Code the video and its audio track."],
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                   response_schema=_VideoModel,
                                                   media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW))
            d = json.loads(r.text)
            return dict(clean(VIDEO_FEATURES, d), evidence=d.get("evidence", ""))
        except Exception as e:
            m = str(e)
            if attempt < 5 and any(k in m for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE")):
                time.sleep(min(90, 10 * 2 ** attempt))
            else:
                raise
        finally:
            if up is not None:
                try: client.files.delete(name=up.name)
                except Exception: pass


def class_probs(bundle, rows):
    import pandas as pd
    X = np.array([[r.get(f, 0) for f in bundle["feats"]] for r in rows], float)
    clf = bundle["model"]
    P = np.zeros((len(X), len(bundle["classes"]))); P[:, clf.classes_] = clf.predict_proba(pd.DataFrame(X, columns=bundle["feats"]))
    return X, P


def explain(bundle, X, top=3):
    feats = bundle["feats"]
    m = X.mean(0)
    out = []
    for i, f in enumerate(feats):
        if f in NEUTRAL: continue
        z = (m[i] - bundle["mean"][f]) / bundle["std"][f]
        sign = -1 if f in NEG else 1
        out.append(dict(feature=f, name=f[2:].replace("_", " "), value=float(m[i]), z=float(z),
                        impact=bundle["importance"][f] * z * sign, what=_DESC.get(f, "").split(". 0=")[0].split(" 0=")[0]))
    out.sort(key=lambda r: r["impact"], reverse=True)
    strengths = [r for r in out if r["impact"] > 0][:top]
    concerns = [r for r in sorted(out, key=lambda r: r["impact"]) if r["impact"] < 0][:top]
    influential = sorted(out, key=lambda r: bundle["importance"][r["feature"]], reverse=True)[:top]
    return strengths, concerns, influential


def analyze(path, client, bundle, max_chunks=8, progress=lambda f, msg: None, extract_fn=None):
    extract_fn = extract_fn or (lambda p: gemini_features(client, bundle["video_model"], p))
    dur = duration(path)
    chunks = pick(plan_chunks(dur, bundle["chunk_sec"]), max_chunks)
    if not chunks:
        raise RuntimeError("video is too short to analyse")
    rows, tmp = [], Path(tempfile.mkdtemp())
    try:
        for n, (idx, s, e, pos) in enumerate(chunks):
            progress(n / len(chunks), f"Analysing minutes {int(s//60)}-{int(e//60)} ({n+1}/{len(chunks)})")
            out = tmp / f"{idx:03d}.mp4"
            cut(path, s, e, out)
            feats = extract_fn(out)
            out.unlink(missing_ok=True)
            rows.append(dict(feats, pos=pos, start=s, end=e, **{f"v_{k}": v for k, v in feats.items() if k != "evidence"}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    X, P = class_probs(bundle, rows)
    mean_p = P.mean(0)
    k = int(mean_p.argmax())
    strengths, concerns, influential = explain(bundle, X)
    progress(1.0, "Done")
    return dict(label=bundle["classes"][k], confidence=float(mean_p[k]), probs=dict(zip(bundle["classes"], map(float, mean_p))),
                agreement=float((P.argmax(1) == k).mean()), expected_level=float(mean_p @ np.arange(len(mean_p))),
                strengths=strengths, concerns=concerns, influential=influential, duration=dur,
                chunks=[dict(start=r["start"], end=r["end"], pred=bundle["classes"][int(p.argmax())], conf=float(p.max()),
                             evidence=r["evidence"]) for r, p in zip(rows, P)])
