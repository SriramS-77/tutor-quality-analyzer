"""
Step 3 - classify chunks (weak / average / good / excellent) and rank them, two ways.

 Execution A  "by_recording"  : train on ONE recording per class, test on the OTHER recordings.
                                (fold k uses the k-th recording of every class as train) -> no leakage.
 Execution B  "interleaved"   : pool all chunks, stratified random 70/30 split, repeated over seeds.
                                Neighbouring chunks of one recording land in both sets -> optimistic.
 If B >> A the model is memorising recordings (overfitting); trust A.

 Ablation over feature sets (text / video / both) and two models (random forest, logistic regression).
 Ranking = expected ordinal score  sum_k p_k * k  (Spearman vs true level).
 Recording-level prediction (A only) = average chunk probabilities over a recording.
 Unlabeled folders (e.g. known-issues) are predicted at the end with the final model.

  python 03_train_eval.py                 # runs both executions
  python 03_train_eval.py --mode by_recording
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

K = len(CLASSES)
LEVEL = {c: i for i, c in enumerate(CLASSES)}


def make_clf(name, seed=0):
    if name == "rf":   # shallow + large leaves: chunks are few and noisy, keep it regularised
        return RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=3, max_features="sqrt",
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))


def load():
    man = pd.read_csv(WORK / "manifest.csv")[["rec_id", "label", "folder"]]
    sets = {}
    t = pd.read_csv(WORK / "features_text.csv") if (WORK / "features_text.csv").exists() else None
    v = pd.read_csv(WORK / "features_video.csv") if (WORK / "features_video.csv").exists() else None
    drop = lambda d: d.drop(columns=[c for c in d.columns if c.endswith("evidence")])
    if t is not None: sets["text"] = drop(t)
    if v is not None: sets["video"] = drop(v)
    if t is not None and v is not None: sets["both"] = drop(t).merge(drop(v).drop(columns="pos"), on=["rec_id", "chunk"])
    out = {}
    for k, d in sets.items():
        d = d.merge(man, on="rec_id")
        d["y"] = d["label"].map(LEVEL).fillna(-1).astype(int)   # -1 = unlabeled
        feats = [c for c in d.columns if c[:2] in ("t_", "v_") or c == "pos"]
        d[feats] = d[feats].fillna(0)
        out[k] = (d, feats)
    return out


def predict_proba(clf, Xtr, ytr, Xte):
    clf.fit(Xtr, ytr)
    p = np.zeros((len(Xte), K)); p[:, clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_] = clf.predict_proba(Xte)
    return p


def metrics(y, p):
    pred = p.argmax(1); score = p @ np.arange(K)
    rho = spearmanr(score, y).correlation if len(set(y)) > 1 else np.nan
    return dict(n=len(y), acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=range(K), average="macro", zero_division=0),
                ordinal_mae=np.abs(pred - y).mean(), spearman=rho,
                confusion=confusion_matrix(y, pred, labels=range(K)).tolist())


def run_by_recording(d, feats, model):
    d = d[d.label.isin(CLASSES)]
    recs = {c: sorted(d[d.label == c].rec_id.unique()) for c in CLASSES}
    n_folds = min(len(v) for v in recs.values())
    res, chunk_p, chunk_y, rec_rows = [], [], [], []
    for k in range(n_folds):
        tr_ids = [recs[c][k] for c in CLASSES]
        tr, te = d[d.rec_id.isin(tr_ids)], d[~d.rec_id.isin(tr_ids)]
        p = predict_proba(make_clf(model, k), tr[feats], tr.y, te[feats])
        m = metrics(te.y.values, p); m["fold"] = k; m["level"] = "chunk"; res.append(m)
        chunk_p.append(p); chunk_y.append(te.y.values)
        g = pd.DataFrame(p, index=te.index).groupby(te.rec_id.values).mean()   # recording level
        yy = te.groupby("rec_id").y.first().loc[g.index].values
        mr = metrics(yy, g.values); mr["fold"] = k; mr["level"] = "recording"; res.append(mr)
        for rid, row in g.iterrows():
            rec_rows.append(dict(fold=k, rec_id=rid, true=CLASSES[te[te.rec_id == rid].y.iloc[0]],
                                 pred=CLASSES[int(row.values.argmax())], exp_score=float(row.values @ np.arange(K))))
    return res, rec_rows


def run_interleaved(d, feats, model, seeds=5, test_size=0.3):
    d = d[d.label.isin(CLASSES)]
    res = []
    for s in range(seeds):
        tr, te = train_test_split(d, test_size=test_size, stratify=d.y, random_state=s)
        p = predict_proba(make_clf(model, s), tr[feats], tr.y, te[feats])
        m = metrics(te.y.values, p); m["fold"] = s; m["level"] = "chunk"; res.append(m)
    return res, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["by_recording", "interleaved", "both"], default="both")
    ap.add_argument("--test-size", type=float, default=0.3, help="interleaved test fraction (0.2 or 0.3)")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    data = load()
    out = WORK / "results"; out.mkdir(exist_ok=True)
    rows, rec_all = [], []
    for fs, (d, feats) in data.items():
        print(f"\n[{fs}] {len(d)} chunks, {len(feats)} features, recordings/class: "
              f"{d[d.label.isin(CLASSES)].groupby('label').rec_id.nunique().to_dict()}")
        for model in ("rf", "lr"):
            for mode in (["by_recording", "interleaved"] if a.mode == "both" else [a.mode]):
                res, rr = (run_by_recording(d, feats, model) if mode == "by_recording"
                           else run_interleaved(d, feats, model, a.seeds, a.test_size))
                for m in res:
                    rows.append(dict(feature_set=fs, model=model, mode=mode, **m))
                rec_all += [dict(feature_set=fs, model=model, **r) for r in rr]

    R = pd.DataFrame(rows)
    R.to_csv(out / "results_all_folds.csv", index=False)
    summ = (R.drop(columns="confusion").groupby(["feature_set", "model", "mode", "level"])
            [["acc", "macro_f1", "ordinal_mae", "spearman"]].mean().round(3))
    summ.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print("\n=== SUMMARY (mean over folds/seeds; chance accuracy = 0.25) ===\n", summ.to_string())
    if a.mode == "both":
        c = summ.xs("chunk", level="level")["macro_f1"].unstack("mode")
        c["gap(interleaved-by_recording)"] = c["interleaved"] - c["by_recording"]
        print("\n=== OVERFIT CHECK (chunk-level macro-F1) ===\n", c.round(3).to_string())
        print("A large positive gap = the model recognises recordings, not quality. Trust by_recording.")
    if rec_all:
        pd.DataFrame(rec_all).to_csv(out / "recording_level_predictions.csv", index=False)

    # ---- final model on ALL labelled recordings -> importances + unlabeled predictions
    fs = "both" if "both" in data else next(iter(data))
    d, feats = data[fs]; lab = d[d.label.isin(CLASSES)]
    rf = make_clf("rf"); rf.fit(lab[feats], lab.y)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv", header=["importance"])
    print(f"\nTop features ({fs}):\n", imp.head(12).round(3).to_string())
    unl = d[~d.label.isin(CLASSES)]
    if len(unl):
        p = pd.DataFrame(rf.predict_proba(unl[feats]), columns=[CLASSES[i] for i in rf.classes_], index=unl.index)
        g = p.groupby(unl.rec_id.values).mean()
        g["predicted"] = g.idxmax(axis=1)
        tech = [c for c in feats if "tech" in c]
        g = g.join(unl.groupby("rec_id")[tech].mean().round(2))
        g.to_csv(out / "unlabeled_predictions.csv")
        print("\nUNLABELED / known-issue recordings (mean chunk probabilities):\n", g.round(2).to_string())


if __name__ == "__main__":
    main()
