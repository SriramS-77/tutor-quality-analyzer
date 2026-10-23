import os, tempfile
from pathlib import Path
import joblib, pandas as pd, streamlit as st
from pipeline import analyze

st.set_page_config(page_title="Tutor Session Quality", page_icon="🎓", layout="centered")
MODEL_PATH = Path(__file__).parent / "model" / "video_rf.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


bundle = load_bundle()
meta = bundle["meta"]
st.title("🎓 Tutor Session Quality")
st.write("Upload one recorded online class (mp4). The server samples 5-minute segments, rates them with a Gemini "
         "vision model, and a random-forest model classifies the lesson as **weak / average / good / excellent**.")
with st.sidebar:
    max_chunks = st.slider("Max 5-min segments to analyse", 3, 14, 8,
                           help="Evenly spread across the lesson. More = slower and uses more free-tier quota.")
    st.caption(f"Video model: `{bundle['video_model']}`")
st.info("🔒 The video is processed on this server, sent to the Google Gemini API for analysis, and deleted afterwards. "
        "Do not upload recordings you are not allowed to share with that service.")

up = st.file_uploader("Class recording", type=["mp4"])
if up and st.button("Analyse recording", type="primary"):
    key = api_key()
    if not key:
        st.error("GEMINI_API_KEY is not configured on the server."); st.stop()
    from google import genai
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(up.getbuffer()); path = f.name
    bar = st.progress(0.0, "Starting...")
    try:
        res = analyze(path, genai.Client(api_key=key), bundle, max_chunks, lambda p, m: bar.progress(min(p, 1.0), m))
        st.session_state["res"] = res
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        os.unlink(path); bar.empty()

res = st.session_state.get("res")
if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", res["label"].upper())
    c2.metric("Confidence", f"{res['confidence']:.0%}")
    c3.metric("Segments agreeing", f"{res['agreement']:.0%}")
    if res["confidence"] < 0.5:
        st.warning("Low confidence: the class probabilities are close together. Treat this as a rough indication only.")
    st.bar_chart(pd.Series(res["probs"], name="probability"))

    def show(title, items, good):
        st.subheader(title)
        if not items: st.write("Nothing notable."); return
        for r in items:
            val = f"{r['value']:.2f}"
            st.markdown(f"- **{r['name'].capitalize()}** (avg {val}; {'above' if r['z'] > 0 else 'below'} the typical "
                        f"training level) — {r['what']}")

    show("✅ What pushed the result up", res["strengths"], True)
    show("⚠️ What pulled it down", res["concerns"], False)
    st.subheader("Most influential signals in the model")
    st.write(", ".join(f"**{r['name']}**" for r in res["influential"]))

    st.subheader("Segment-by-segment")
    st.dataframe(pd.DataFrame([dict(minutes=f"{int(c['start']//60)}–{int(c['end']//60)}", predicted=c["pred"],
                                    confidence=f"{c['conf']:.0%}", observations=c["evidence"]) for c in res["chunks"]]),
                 hide_index=True, use_container_width=True)
    st.caption(f"⚠️ Reliability: this model was trained on only {meta['n_recordings']} labelled recordings. On held-out "
               f"recordings it classified about {meta['recording_acc']:.0%} of whole lessons correctly (chance = "
               f"{meta['chance']:.0%}); that estimate is itself very noisy. The 'reasons' describe how the lesson's rated "
               f"features compare with the training data, weighted by feature importance. Use as decision support, "
               f"not as a verdict.")
