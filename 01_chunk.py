"""
Step 1 - split every mp4 and every vtt INDEPENDENTLY into CHUNK_SEC windows.

 * VTT  -> work/chunks/text/<rec>/<idx>.json   (speaker-anonymised cues starting in the window)
 * MP4  -> work/chunks/video/<rec>/<idx>.mp4   (360p, 1 fps, mono 16 kHz audio -> small & cheap to upload)
 * work/manifest.csv                           (one row per recording)
Resumable: existing outputs are skipped.
"""
import csv, json, math, subprocess, sys
from common import *


def duration(mp4):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", mp4], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def chunk_video(mp4, outdir):
    if outdir.exists():                                     # finished earlier (we rename only when complete)
        return
    tmp = outdir.with_name(outdir.name + ".tmp")
    if tmp.exists():
        for f in tmp.iterdir(): f.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-map", "0:v:0", "-map", "0:a:0?",
           "-vf", "scale=-2:360,fps=1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "33",
           "-force_key_frames", f"expr:gte(t,n_forced*{CHUNK_SEC})",
           "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "16000", "-sn", "-dn",
           "-f", "segment", "-segment_time", str(CHUNK_SEC), "-reset_timestamps", "1",
           str(tmp / "%03d.mp4")]
    subprocess.run(cmd, check=True)
    tmp.rename(outdir)


def main():
    (WORK / "chunks").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in discover():
        rid = r["rec_id"]
        cues = anonymise(parse_vtt(r["vtt"])) if r["vtt"] else []
        dur = duration(r["mp4"]) if r["mp4"] else 0
        if not dur and cues:
            dur = max(c[1] for c in cues)
        n_all = math.ceil(dur / CHUNK_SEC)
        keep = [i for i in range(n_all) if min((i + 1) * CHUNK_SEC, dur) - i * CHUNK_SEC >= MIN_CHUNK_SEC]
        print(f"{rid}: {dur/60:.1f} min, {len(cues)} cues, {len(keep)} chunks  <- {r['folder']}")

        tdir = WORK / "chunks" / "text" / rid
        tdir.mkdir(parents=True, exist_ok=True)
        for i in keep:
            s, e = i * CHUNK_SEC, min((i + 1) * CHUNK_SEC, dur)
            part = [c for c in cues if s <= c[0] < e]
            pos = i / max(1, keep[-1])
            json.dump(dict(rec_id=rid, chunk=i, start=s, end=e, pos=pos, cues=part),
                      open(tdir / f"{i:03d}.json", "w"))

        vdir = WORK / "chunks" / "video" / rid
        chunk_video(r["mp4"], vdir)
        for f in vdir.glob("*.mp4"):                       # drop tail chunks that are too short
            if int(f.stem) not in keep:
                f.unlink()
        rows.append(dict(r, duration=round(dur, 1), n_chunks=len(keep), last_chunk=keep[-1] if keep else 0))

    with open(WORK / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {WORK/'manifest.csv'} ({len(rows)} recordings)")


if __name__ == "__main__":
    main()
