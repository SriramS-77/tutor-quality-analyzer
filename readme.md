# AI Interaction History: Tutor Session Quality Analyzer

A condensed, chronological account of how the project was ideated and built with Claude (Claude Code / Sonnet) over
roughly one evening. It is a curated summary, not a verbatim transcript. Personal details, tutor names and credentials
are omitted. Hosting and deployment debugging is deliberately excluded.

## 1. Understanding the brief and choosing a strategy

**Request:** "I have 2 recordings per class, an mp4 (1 hour) and a .vtt for each. What approach should I use?"

**What Claude advised**
- With about 8 labelled recordings, nothing can be trained end to end. Use a **hybrid system**: measurable features from the
  transcript, audio and video, plus an LLM scoring against a fixed rubric. Use the labelled samples to calibrate, not to train.
- Check the data rules first: the brief restricts which services may process recordings of children. Prefer local
  processing and send only transcript text (names stripped) to an LLM until the organisers confirm which APIs are allowed.
- Study one excellent and one poor class first to see what actually differs.
- Candidate signals: talk ratio, questions and wait time, student response length, silences, tech-trouble phrases, praise
  and corrections. Warn that "more teacher talk is bad" is not automatically true, and that video quality or camera-on
  rates can be misleading signals.
- Plan: chunked rubric scoring with timestamped evidence, a calibrated final label, a simple Streamlit or Gradio app, and an
  early deployment test.
- Asked whether the VTT has speaker labels, since that decides how much work diarization would need.

## 2. Designing the experiment

**Request (paraphrased):** Write scripts that split the mp4 and vtt independently into chunks, extract features using
free-tier Gemini Flash models (a cheap text model and a multimodal video model, with separate prompts per modality),
then train a classifier such as a random forest on chunks. Features must be generic metrics, not recording-specific
details like background colour, so they cannot be used to overfit after seeing a recording once. Run two executions:
(A) train and test on separate recordings, (B) interleaved 70/30 or 80/20 chunk split, to detect overfitting. Folder
layout: `class_recordings/<class>/`, with `weak` instead of `poor`, a `demo` folder to skip, and an open question about
how to use the "known issues" recordings.

**Decisions made**
- **Chunk size:** 5 minutes. Gives about 12 chunks per hour, is long enough to contain interaction, and short enough for
  a free-tier call. A 45-second minimum keeps the closing segment where recap and homework occur.
- **Label for known-issue recordings:** the feedback PDF gave per-recording comments grouped as excellent, good,
  average, weak, and a separate "known quality issues" group with no grade. Those recordings are treated as **unlabeled**:
  never trained or scored on, only predicted at the end as a qualitative check. The reviewers' text is not used as a feature,
  since it would not exist for a new recording.
- **Anti-overfitting feature design:** every feature is a bounded number with an anchored scale (0-4 ratings, counts, flags,
  0-1 fractions). Prompts never reveal the label and never ask for an overall judgement. Only numeric fields reach the
  classifier; the free-text evidence sentence is kept for explanations.
- **Models:** left configurable through environment variables, because the exact free-tier model IDs were uncertain. A small
  script lists the models a given key can call.
- **Privacy:** speaker names replaced with `TUTOR` / `STUDENT_n` before sending transcripts; video cannot be anonymised, so
  the free-tier data-use caveat was flagged for confirmation with the organisers.
- **Two modalities, separate features:** transcript features plus deterministic transcript statistics (talk share,
  silences, student turns); video features from a multimodal model that also hears the audio.

## 3. Building and smoke-testing the pipeline

Claude wrote `common.py`, `feature_spec.py`, `01_chunk.py`, `02_extract_features.py`, `03_train_eval.py`, then tested the
parts that do not need an API key on synthetic recordings (fake VTTs, generated mp4s and fake cached model outputs).

**Bugs found during testing and fixed**
- An interrupted ffmpeg run left a partial chunk folder that a rerun would have treated as finished. Chunking now writes to
  a temporary folder and renames it on completion.
- A label column became floating point when unlabeled recordings were present, breaking classifier indexing. Fixed by
  casting with a sentinel for unlabeled rows.
- A synthetic check confirmed the overfit diagnostic behaves as intended: interleaved scores exceeded by-recording scores
  when recordings had individual quirks.

Setup note: the first real run failed because `ffprobe` was not installed on Windows. The fix was installing ffmpeg via
winget and restarting the terminal so the PATH refreshed.

## 4. First real results and analysis

Results showed by-recording scores near chance (chunk macro-F1 about 0.24-0.39, chance 0.25) and a large interleaved gap
(about 0.25-0.30 for text and combined features). Claude's analysis:
- By-recording is the honest setting; the interleaved numbers mostly reflect recognising recordings.
- Recording-level Spearman values of 0.6-0.9 rested on only 4 recordings per fold and should not be quoted as evidence.
- Feature importances were computed on a model fitted to all data, so they are not evidence either.
- Chunk labels are noisy by construction, because the feedback is about whole-lesson behaviours.

**Question:** should we add features that look important, or try XGBoost?

**Answer:** no to both. With 8 recordings the limit is data, not model capacity, and choosing features after seeing which
ones score is selection leakage. Recommended recording-level aggregation, leave-one-recording-out validation, regression
on the ordinal level, and a small feature list fixed in advance from the rubric.

## 5. Recording-level, leave-one-out validation

Claude wrote `04_recording_level.py`: per-recording aggregates (means, with "any chunk" for flags), leave-one-recording-out
predictions with equal-weight z-score, ridge and baseline models, a feature list fixed beforehand from the feedback themes,
pairwise ordering accuracy, and a label-permutation p-value. Synthetic tests passed; a misleading baseline Spearman of -1
(a leave-one-out artefact) was blanked out.

**Real results:** no feature set beat chance (all permutation p-values 0.34 or higher; negative correlations are
expected under no signal with leave-one-out). One recording's score was an extreme outlier, showing instability. The
known-issue recordings were not detected by the video features (tech disruption 0.0), likely because 360p at 1 fps hides
freezes. Claude's conclusion: report it honestly as weak signal at n=8, and suggest the next real step: turn the
reviewers' comments into per-item checklist labels and validate features against those (not built).

## 6. Choosing the model for the app

**Request:** use the best video-only model (it did best on holdout) and build a simple single-page web app where a user
uploads one mp4 and gets a class, confidence and reasons.

**Decisions**
- Video-only random forest: best by-recording chunk score and smallest overfit gap among the options.
- `05_export_model.py` retrains on all 8 labelled recordings, stores the model, training feature statistics and importances,
  and records a holdout estimate, so the app can display honest reliability. It also writes a pinned `requirements.txt`
  so the saved model loads with the same library versions.
- App pipeline: cut evenly spaced 5-minute segments from the upload, extract features with the same model and prompt as
  training, average chunk probabilities into the lesson-level class, then derive reasons by comparing feature averages with
  the training data weighted by importance.
- Used `imageio-ffmpeg` so the app needs no system ffmpeg, a slider to cap segments for free-tier quota, deletion of the
  uploaded file after processing, and an on-page privacy and reliability notice.
- Tested with the Gemini call stubbed and a headless Streamlit page load. A test model trained on synthetic data was deleted
  before packaging so only a model built from real features is deployed.

**Question:** "What is the recording-level accuracy for this video-only model?"

**Honest answer:** 50% (4 of 8 whole-recording predictions, chance 25%), but the probability of 4 or more of 8 by
guessing is about 11%, and the stricter leave-one-recording-out run showed video models at or below chance. The app
footer reports the 50% figure; the recommendation was to show the leave-one-out result alongside it.

## 7. Outputs

`00_list_models.py`, `01_chunk.py`, `02_extract_features.py`, `03_train_eval.py`, `04_recording_level.py`,
`05_export_model.py`, the `app/` Streamlit app, and this README and history.

## Lessons and honest limits

- The approach was sound for exploration, but 8 recordings cannot support a reliable four-way classifier.
- Claude repeatedly steered away from choices that would have inflated apparent performance (interleaved splits,
  post-hoc feature selection, a heavier model), and surfaced weak results instead of tuning until they looked good.
- Remaining work: more labelled recordings, checklist-based labels from the written feedback, higher frame rate for
  detecting freezes, and confirming with the organisers that the Gemini API is permitted for these recordings.
  