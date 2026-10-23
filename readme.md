# Tutor Session Quality Analyzer

Assesses the quality of a recorded online class for children and classifies it as **weak / average / good / excellent**,
with a confidence value and the observations behind the decision.

**Live app:** https://tutor-quality-analyzer--sriramsun.replit.app

Upload one `.mp4` recording. The server samples 5-minute segments, rates each with a Gemini multimodal model, and a
random-forest classifier combines the segment results into a class for the whole lesson. Expect roughly 2 minutes of
processing for the default 8 segments (the free Gemini tier is rate-limited).

> **Reliability note.** This is a prototype trained on 8 labelled recordings. Held-out accuracy is about 50% on 8
> whole-recording predictions (chance is 25%), which is not statistically distinguishable from luck, and a stricter
> leave-one-recording-out test showed no evidence of signal. See [Results and limitations](#results-and-limitations).
> Use the output as decision support, not as a verdict.

---

## Methodology

### 1. Problem framing
The brief gives a handful of recordings (an `.mp4` and a `.vtt` transcript per class) at four quality levels. Too few to
train a model end to end, so the approach is **extract generic, behaviour-level measurements with an LLM, then fit a small,
regularised classifier on them**.

### 2. Chunking
Each recording is split independently into 5-minute windows (a 45-second minimum keeps the short closing segment, where
recap and homework usually happen):
- **Video:** re-encoded to 360p, 1 fps, mono 16 kHz audio so uploads are small and cheap.
- **Transcript:** cues are assigned by start time, and speaker names are replaced by `TUTOR` / `STUDENT_n`.

### 3. Feature extraction (separate prompt and model per modality)
- **Text** (Gemini Flash-Lite plus deterministic transcript statistics): clarity, understanding checks, learner
  participation, individual attention, warmth, feedback quality, pacing, classroom management, counts of questions /
  praise / checks, and flags for greeting, learner introductions, recap and homework.
- **Video** (Gemini Flash, which sees video and audio): tutor visibility and face clarity, lighting, stability, tidiness
  of surroundings, audio clarity, noise, expressiveness, vocal energy, gestures, eye contact, visual aids, interactive
  activities, learner engagement, tech disruption, speaking fractions.

Design rules to stop the downstream model memorising one recording:
- every feature is a bounded number (0-4 rating, count, flag or 0-1 fraction) with an **anchored scale**;
- no feature refers to colours, names, subject or room details;
- the prompt never reveals the class label and never asks for an overall quality verdict;
- only the numeric fields reach the classifier. The free-text "evidence" sentence is for humans.

### 4. Classification
Random forest (400 trees, depth 4, min leaf 3, balanced class weights), deliberately shallow because chunks are few and
noisy. A logistic regression is used as a comparison. At inference, chunk probabilities are averaged over the lesson to
give the class and confidence. The share of segments that agree with the final class is also shown.

### 5. Explanations
Each feature's lesson average is compared with the training distribution, weighted by the forest's feature
importance. This produces "what pushed the result up", "what pulled it down", and a segment-by-segment table with the
vision model's observations. These describe the rated features, not the forest's exact internal decision path.

### 6. Evaluation (to check for overfitting)
Two executions, run for text-only, video-only and combined features:
- **By recording:** train on one recording per class, test on the remaining recordings. No recording appears on both sides.
- **Interleaved:** pool all chunks and split 70/30 at random. Neighbouring chunks of the same recording land on both sides, so
  this is optimistic. A large gap between the two executions means the model recognises recordings rather than quality.
- **Leave-one-recording-out** (recording-level): train on 7, predict the 8th, with a label-permutation p-value.

Known-quality-issue recordings have no quality grade, so they are never used for training or scoring. They are
predicted afterwards as a qualitative check.

### Results and limitations

| Setting (chunk-level macro-F1) | Text | Video | Both |
|---|---|---|---|
| By recording (no leakage) | ~0.24-0.28 | ~0.29-0.39 | ~0.28-0.30 |
| Interleaved (leaky) | ~0.52-0.54 | ~0.44-0.46 | ~0.53-0.60 |

- Chance is about 0.25. Text and combined models showed a large interleaved-vs-by-recording gap (about 0.25-0.30), a
  clear sign of memorising recordings.
- Video-only random forest had the smallest gap (about 0.07) and the best by-recording score, which is why it is the model
  in the app. Its recording-level accuracy was 0.5 (4 of 8 predictions), which is not significant.
- In leave-one-recording-out, no feature set beat chance (all permutation p-values 0.34 or higher; video concordance
  0.21-0.29). Features do not yet separate the four grades reliably.
- The video features did not register the known-issue recordings (tech disruption 0.0), partly because 360p at 1 fps hides
  freezes.
- Eight recordings can only detect very large effects, and the labels are subjective, whole-lesson judgements that
  each 5-minute chunk only partly reflects.

**What would help most:** more labelled recordings, turning the reviewers' written feedback into per-item checklist labels
(background, homework, rapport, introductions, energy, camera) and validating features against those, and scoring the
whole recording directly against that checklist.

---

## Run locally

Requires Python 3.11+ and a Gemini API key. The research scripts need `ffmpeg` and `ffprobe` on PATH.

```bash
pip install -r requirements.txt                  # root: google-genai pydantic pandas numpy scikit-learn scipy
export GEMINI_API_KEY=...                        # PowerShell: $env:GEMINI_API_KEY="..."
export DATA_DIR=/path/to/class_recordings        # folders: weak/ average/ good/ excellent/ (+ any unlabeled folder)

python 00_list_models.py                         # which Gemini models your key can call
export TEXT_MODEL=...; export VIDEO_MODEL=...    # optional overrides

python 01_chunk.py                               # split mp4 / vtt into 5-min chunks
python 02_extract_features.py                    # Gemini feature extraction (resumable, cached)
python 03_train_eval.py                          # by-recording vs interleaved evaluation
python 04_recording_level.py                     # leave-one-recording-out, permutation test
python 05_export_model.py --video-model <model used in step 02>   # trains and exports the app model
```

Run the web app:

```bash
cd app
pip install -r requirements.txt                  # generated by step 05 with matching versions
export GEMINI_API_KEY=...
streamlit run app.py
```

Do not commit recordings, the `work/` folder, transcripts or API keys.

## Repository layout

| Path | Purpose |
|---|---|
| `common.py`, `feature_spec.py` | Shared config, VTT parsing and anonymisation, feature definitions and prompt rules |
| `00_list_models.py` ... `05_export_model.py` | The research and export pipeline described above |
| `app/app.py` | Streamlit single-page UI |
| `app/pipeline.py` | Server-side: cut segments, Gemini features, random forest, explanations |
| `app/model/video_rf.joblib` | Exported model bundle (no recording data) |

## Data handling

Recordings are confidential and contain children. Uploaded videos are processed on the server, sent to the Google
Gemini API for analysis, and deleted afterwards. Free-tier Gemini data may be used by Google to improve its products, so
confirm with the organisers that this service is permitted before processing any recording you are not entitled to share
with it. Speaker names are stripped from transcripts before any LLM call, but video cannot be anonymised.
