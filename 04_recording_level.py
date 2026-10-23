"""
Step 4 - recording-level scoring with leave-one-recording-out (LORO) validation.

Why: with ~8 labelled recordings the honest unit is the RECORDING, and the label is a whole-lesson judgement.
 * chunk features -> per-recording aggregates (mean; flags = "any chunk")
 * LORO: train on 7 recordings, predict the 8th, repeat -> 8 leak-free predictions
 * models predict a continuous quality level 0..3 (weak..excellent), then round to a class
     zscore_equal : equal-weight signed z-score of an A-PRIORI rubric feature list (PRIOR below,
                    chosen from the feedback themes BEFORE seeing any results), mapped to a level
                    with a 1-D linear fit
     ridge_prior  : ridge regression on the same a-priori features
     ridge_all    : ridge on every aggregated feature (contrast: more features, same 7 samples)
     mean_baseline: always predicts the training mean (reference)
 * metrics: Spearman, pairwise ordering accuracy (concordance), MAE, accuracy, within-1 accuracy,
            plus a label-permutation p-value for concordance.
 * known-issue / unlabeled recordings are predicted at the end, with a tech-feature comparison table.

  python 04_recording_level.py
  python 04_recording_level.py --n-perm 500 --alpha 10
DON'T edit PRIOR after looking at the results - that re-introduces the selection leakage this step avoids.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from common import *
from feature_spec import TEXT_FEATURES, VIDEO_FEATURES

K = len(CLASSES); LEVEL = {c: i for i, c in enumerate(CLASSES)}
FLAGS = {f"t_{n}" for n, k, _ in TEXT_FEATURES if k == "flag"} | {f"v_{n}" for n, k, _ in VIDEO_FEATURES if k == "flag"}

# (column, sign)  +1 = higher is better, -1 = higher is worse.  Fixed in advance from the rubric/feedback themes.
PRIOR = {
    "text": [("t_explanation_clarity", 1), ("t_understanding_checking", 1), ("t_learner_participation", 1),
             ("t_individual_attention", 1), ("t_encouragement_warmth", 1), ("t_tech_disruption", -1),
             ("t_greeting_or_rapport", 1), ("t_learner_introductions", 1), ("t_recap_or_closing", 1),
             ("t_practice_or_homework_assigned", 1)],
    "video": [("v_tutor_on_camera_frac", 1), ("v_tutor_face_clarity", 1), ("v_tutor_expressiveness", 1),
              ("v_tutor_vocal_energy", 1), ("v_backdrop_neatness", 1), ("v_lighting_quality", 1),
              ("v_audio_clarity", 1), ("v_interactive_activity", 1), ("v_tech_disruption", -1)],
}


def load_recordings():
    man = pd.read_csv(WORK / "manifest.csv")[["rec_id", "label"]]
    parts = []
    for fn in ("features_text.csv", "features_video.csv"):
        p = WORK / fn
        if p.exists():
            d = pd.read_csv(p)
            d = d.drop(columns=[c for c in d.columns if c.endswith("evidence")] + ["chunk", "pos"], errors="ignore")
            parts.append(d.groupby("rec_id").agg({c: ("max" if c in FLAGS else "mean") for c in d.columns if c != "rec_id"}))
    R = pd.concat(parts, axis=1).reset_index().merge(man, on="rec_id")
    R["y"] = R.label.map(LEVEL)                         # NaN for unlabeled
    return R


def feature_sets(R, alpha):
    cols = {"text": [c for c in R if c.startswith("t_")], "video": [c for c in R if c.startswith("v_")]}
    cols["both"] = cols["text"] + cols["video"]
    out = {}
    for fs in ("text", "video", "both"):
        if not cols[fs]: continue
        prior = [(c, s) for src in (["text", "video"] if fs == "both" else [fs]) for c, s in PRIOR[src] if c in R]
        if prior: out[fs] = dict(prior=prior, all=cols[fs])
    return out


def _z(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std(ddof=0).replace(0, 1)
    return (tr[cols] - mu) / sd, (te[cols] - mu) / sd


def predict(model, tr, te, spec, alpha):
    y = tr.y.values.astype(float)
    if model == "mean_baseline":
        return np.full(len(te), y.mean())
    if model == "zscore_equal":
        cols = [c for c, _ in spec["prior"]]; sg = np.array([s for _, s in spec["prior"]])
        ztr, zte = _z(tr, te, cols)
        s_tr, s_te = (ztr.values * sg).mean(1), (zte.values * sg).mean(1)
        lr = LinearRegression().fit(s_tr.reshape(-1, 1), y)
        return lr.predict(s_te.reshape(-1, 1))
    cols = [c for c, _ in spec["prior"]] if model == "ridge_prior" else spec["all"]
    ztr, zte = _z(tr, te, cols)
    a = alpha * (len(cols) / 10 if model == "ridge_all" else 1)
    return Ridge(alpha=a).fit(ztr.fillna(0), y).predict(zte.fillna(0))


def loro(model, L, spec, alpha, y_override=None):
    L = L.copy()
    if y_override is not None: L["y"] = y_override
    preds = np.zeros(len(L))
    for i in range(len(L)):
        tr, te = L.drop(L.index[i]), L.iloc[[i]]
        preds[i] = predict(model, tr, te, spec, alpha)[0]
    return preds, L.y.values.astype(float)


def concordance(p, y):
    num = den = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] != y[j]:
                den += 1
                d = (p[j] - p[i]) * np.sign(y[j] - y[i])
                num += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
    return num / den if den else np.nan


def score(p, y):
    c = np.clip(np.round(p), 0, K - 1)
    return dict(spearman=spearmanr(p, y).correlation if np.std(p) > 0 else np.nan, concordance=concordance(p, y),
                mae=np.abs(c - y).mean(), acc=(c == y).mean(), within1=(np.abs(c - y) <= 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge strength (larger = simpler)")
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()
    R = load_recordings()
    L = R[R.label.isin(CLASSES)].reset_index(drop=True)
    U = R[~R.label.isin(CLASSES)].reset_index(drop=True)
    sets = feature_sets(R, a.alpha)
    models = ["zscore_equal", "ridge_prior", "ridge_all", "mean_baseline"]
    rng = np.random.default_rng(0)
    out = WORK / "results"; out.mkdir(exist_ok=True)
    print(f"{len(L)} labelled recordings ({L.label.value_counts().to_dict()}), {len(U)} unlabeled")
    print("Reference: random guessing -> MAE ~1.25, acc 0.25, concordance 0.5, spearman 0\n")

    rows, pred_rows = [], []
    for fs, spec in sets.items():
        for m in models:
            p, y = loro(m, L, spec, a.alpha)
            s = score(p, y)
            null = []
            if m != "mean_baseline":
                for _ in range(a.n_perm):
                    pp, yy = loro(m, L, spec, a.alpha, rng.permutation(L.y.values))
                    null.append(concordance(pp, yy))
                s["perm_p"] = (1 + np.sum(np.array(null) >= s["concordance"] - 1e-9)) / (1 + len(null))
            if m == "mean_baseline":                         # rank metrics are meaningless (LORO artefact)
                s["spearman"] = s["concordance"] = np.nan
            rows.append(dict(feature_set=fs, model=m, **s))
            for rid, lab, pv in zip(L.rec_id, L.label, p):
                pred_rows.append(dict(feature_set=fs, model=m, rec_id=rid, true=lab,
                                      score=round(float(pv), 2), pred=CLASSES[int(np.clip(round(pv), 0, K - 1))]))
    T = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 200)
    print("=== LEAVE-ONE-RECORDING-OUT RESULTS ===\n", T.to_string(index=False))
    print("\nperm_p = chance of getting this concordance (or better) with shuffled labels. With 8 recordings "
          "treat p>0.1 as 'no evidence'.")
    T.to_csv(out / "loro_summary.csv", index=False)
    P = pd.DataFrame(pred_rows); P.to_csv(out / "loro_predictions.csv", index=False)
    show = P[(P.feature_set == "both") & (P.model == "zscore_equal")] if "both" in sets else P.head(0)
    if len(show):
        print("\nPer-recording LORO predictions (both / zscore_equal):\n", show[["rec_id", "true", "score", "pred"]].to_string(index=False))

    # ---- unlabeled recordings, predicted by models fit on all labelled recordings
    if len(U):
        up = []
        for fs, spec in sets.items():
            for m in models[:3]:
                pv = predict(m, L, U, spec, a.alpha)
                for rid, v in zip(U.rec_id, pv):
                    up.append(dict(feature_set=fs, model=m, rec_id=rid, score=round(float(v), 2),
                                   pred=CLASSES[int(np.clip(round(v), 0, K - 1))]))
        UP = pd.DataFrame(up); UP.to_csv(out / "unlabeled_recording_predictions.csv", index=False)
        print("\n=== UNLABELED / known-issue recordings ===\n", UP.pivot_table(index="rec_id", columns=["feature_set", "model"],
              values="score").round(2).to_string())

    # ---- confound check: are tech-issue features higher for the known-issue recordings?
    tech = [c for c in R if "tech" in c]
    C = R[["rec_id", "label"] + tech].copy()
    C["label"] = pd.Categorical(C.label, CLASSES + sorted(set(C.label) - set(CLASSES)), ordered=True)
    print("\n=== TECH-ISSUE FEATURES PER RECORDING (known-issue recordings should stand out) ===\n",
          C.sort_values(["label", "rec_id"]).round(2).to_string(index=False))
    R.to_csv(out / "recording_features.csv", index=False)


if __name__ == "__main__":
    main()