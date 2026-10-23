"""
Step 5 - train the final VIDEO-ONLY random forest on ALL labelled recordings and export it for the web app.

  set VIDEO_MODEL to the SAME Gemini model you used in step 02 (or pass --video-model); the app must extract
  features with the identical model/prompt, otherwise the feature distribution shifts.

  python 05_export_model.py --video-model gemini-2.5-flash
Writes: app/model/video_rf.joblib, app/requirements.txt, app/feature_spec.py
"""
import argparse, importlib.metadata as md, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from common import *

APP = Path(__file__).parent / "app"
K = len(CLASSES); LEVEL = {c: i for i, c in enumerate(CLASSES)}


def make_rf(seed=0):                      # identical to 03_train_eval.py
    return RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=3, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


def proba(clf, X):
    p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video-model", default=VIDEO_MODEL); a = ap.parse_args()
    d = pd.read_csv(WORK / "features_video.csv")
    d = d.drop(columns=[c for c in d.columns if c.endswith("evidence")])
    d = d.merge(pd.read_csv(WORK / "manifest.csv")[["rec_id", "label"]], on="rec_id")
    d = d[d.label.isin(CLASSES)].copy(); d["y"] = d.label.map(LEVEL).astype(int)
    feats = ["pos"] + [c for c in d.columns if c.startswith("v_")]
    d[feats] = d[feats].fillna(0)

    # honest hold-out estimate: Execution A (train 1 recording per class, test on the rest), same as step 03
    recs = {c: sorted(d[d.label == c].rec_id.unique()) for c in CLASSES}
    f1s, accs, racc = [], [], []
    for k in range(min(len(v) for v in recs.values())):
        tr_ids = [recs[c][k] for c in CLASSES]
        tr, te = d[d.rec_id.isin(tr_ids)], d[~d.rec_id.isin(tr_ids)]
        p = proba(make_rf(k).fit(tr[feats], tr.y), te[feats])
        f1s.append(f1_score(te.y, p.argmax(1), labels=range(K), average="macro", zero_division=0))
        accs.append(accuracy_score(te.y, p.argmax(1)))
        g = pd.DataFrame(p, index=te.index).groupby(te.rec_id.values).mean()
        racc.append(accuracy_score(te.groupby("rec_id").y.first().loc[g.index], g.values.argmax(1)))
    meta = dict(chunk_macro_f1=float(np.mean(f1s)), chunk_acc=float(np.mean(accs)), recording_acc=float(np.mean(racc)),
                n_recordings=int(d.rec_id.nunique()), n_chunks=len(d), chance=1 / K)
    print("hold-out estimate:", {k: round(v, 3) if isinstance(v, float) else v for k, v in meta.items()})

    clf = make_rf().fit(d[feats], d.y)
    bundle = dict(model=clf, feats=feats, mean=d[feats].mean().to_dict(), std=d[feats].std(ddof=0).replace(0, 1).to_dict(),
                  importance=dict(zip(feats, map(float, clf.feature_importances_))), classes=CLASSES,
                  video_model=a.video_model, chunk_sec=CHUNK_SEC, meta=meta)
    (APP / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, APP / "model" / "video_rf.joblib", compress=3)
    shutil.copy(Path(__file__).parent / "feature_spec.py", APP / "feature_spec.py")
    v = lambda p: md.version(p)
    (APP / "requirements.txt").write_text(
        f"streamlit>=1.40\ngoogle-genai>=1.0.0\npydantic>=2.0\nimageio-ffmpeg\njoblib=={v('joblib')}\n"
        f"numpy=={v('numpy')}\npandas\nscikit-learn=={v('scikit-learn')}\n")
    print(f"exported model (video features by '{a.video_model}') -> {APP/'model'/'video_rf.joblib'}")


if __name__ == "__main__":
    main()
