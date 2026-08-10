# =====================================================
# APP.PY — A-SASS Speech Therapy (Therapist vs Patient)
# =====================================================

import streamlit as st
import tempfile
import subprocess
import os
import cv2
import librosa
import soundfile as sf
import whisper
import numpy as np
import difflib
import torch

from pathlib import Path
from jiwer import wer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="A-SASS Speech Therapy", page_icon="🧠", layout="wide")
st.title("🧠 A-SASS Speech Therapy System")
st.markdown(
    """
    Upload a **therapist reference video** and compare it with the **patient's session**
    using speech recognition, phoneme analysis, and facial articulation scoring.
    """
)

# =====================================================
# SESSION STATE INIT
# =====================================================

def _init_state():
    defaults = {
        "therapist_saved": False,
        "patient_saved":   False,
        "results":         None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# =====================================================
# TEMP PATHS
# =====================================================

TEMP_DIR             = tempfile.gettempdir()
THERAPIST_VIDEO_PATH = os.path.join(TEMP_DIR, "therapist_video.mp4")
PATIENT_VIDEO_PATH   = os.path.join(TEMP_DIR, "patient_video.mp4")

# =====================================================
# HELPERS
# =====================================================

def extract_audio(video_path: str, audio_path: str):
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000",
         "-af", "loudnorm", "-vn", audio_path, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def safe_wer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if ref == "" and hyp == "": return 0.0
    if ref == "" or  hyp == "": return 1.0
    try:    return float(wer(ref, hyp))
    except: return 1.0

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (frames / fps) if fps else 0.0

def record_video(output_path: str, duration: float):
    cap    = cv2.VideoCapture(0)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))
    st.info(f"Recording {duration:.2f} seconds...")
    for _ in range(int(duration * 20)):
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
    cap.release(); writer.release()
    st.success("Recording Completed")

def word_diff_html(ref_text: str, hyp_text: str):
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    matcher   = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    M  = "background:#d4edda;color:#155724;border-radius:4px;padding:2px 5px;margin:2px;display:inline-block;"
    RM = "background:#f8d7da;color:#721c24;border-radius:4px;padding:2px 5px;margin:2px;display:inline-block;text-decoration:line-through;"
    HM = "background:#fff3cd;color:#856404;border-radius:4px;padding:2px 5px;margin:2px;display:inline-block;"
    SR = "background:#f8d7da;color:#721c24;border-radius:4px;padding:2px 5px;margin:2px;display:inline-block;"
    SH = "background:#fff3cd;color:#856404;border-radius:4px;padding:2px 5px;margin:2px;display:inline-block;"
    rp, hp = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for w in ref_words[i1:i2]: rp.append(f'<span style="{M}">{w}</span>')
            for w in hyp_words[j1:j2]: hp.append(f'<span style="{M}">{w}</span>')
        elif op == "delete":
            for w in ref_words[i1:i2]: rp.append(f'<span style="{RM}">{w}</span>')
        elif op == "insert":
            for w in hyp_words[j1:j2]: hp.append(f'<span style="{HM}">{w}</span>')
        elif op == "replace":
            for w in ref_words[i1:i2]: rp.append(f'<span style="{SR}">{w}</span>')
            for w in hyp_words[j1:j2]: hp.append(f'<span style="{SH}">{w}</span>')
    wrap = '<div style="line-height:2.4;font-size:1.1rem;">{}</div>'
    return wrap.format(" ".join(rp)), wrap.format(" ".join(hp))

def diff_legend_html():
    return """
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;font-size:0.9rem;">
        <span style="background:#d4edda;color:#155724;border-radius:4px;padding:2px 8px;">✓ Correct</span>
        <span style="background:#f8d7da;color:#721c24;border-radius:4px;padding:2px 8px;">✗ Missing / Wrong</span>
        <span style="background:#fff3cd;color:#856404;border-radius:4px;padding:2px 8px;">⚠ Extra / Substituted</span>
    </div>
    """

def transcribe(model, audio_path: str, lang: str, temp_dir: str) -> dict:
    if not os.path.exists(audio_path):
        return {"text": "", "raw_runs": [], "romanized": False}

    lang_code = "hi" if lang.lower() == "hindi" else "en"
    result = model.transcribe(audio_path, language=lang_code)
    text_out = result.get("text", "").strip()

    is_romanized = False
    if lang_code == "hi" and text_out:
        ascii_count = sum(1 for c in text_out if ord(c) < 128)
        if ascii_count / len(text_out) > 0.8:
            is_romanized = True

    return {
        "text": text_out,
        "raw_runs": [text_out],
        "romanized": is_romanized
    }

# =====================================================
# AVSR MODEL
# =====================================================

WHISPER_MODEL_OPTIONS = {
    "medium (recommended for Hindi)": "medium",
    "small (faster, less accurate)": "small",
    "large-v2 (best accuracy, slow)": "large-v2",
}

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    if device == "cuda":
        torch.cuda.empty_cache()
    return model

# =====================================================
# PATIENT INFO + MODEL
# =====================================================

st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    patient_name = st.text_input("👤 Patient Name")

with c2:
    language = st.selectbox("🌐 Language", ["Hindi", "English"])

with c3:
    selected_model_label = st.selectbox(
        "🤖 AVSR Model",
        list(WHISPER_MODEL_OPTIONS.keys()),
        index=0,
        help="'base' gives wrong Hindi results. Use medium or large-v2."
    )

selected_model = WHISPER_MODEL_OPTIONS[selected_model_label]

if language == "Hindi" and selected_model in ["small"]:
    st.warning("⚠️ 'small' model may romanize Hindi. Medium or large-v2 is strongly recommended.")

# =====================================================
# STEP 1 — UPLOAD THERAPIST REFERENCE VIDEO
# =====================================================

st.divider()
st.header("Step 1: Upload Therapist Reference Video")

therapist_file = st.file_uploader(
    "Upload therapist video", type=["mp4", "mov", "avi", "mkv"],
    key="therapist_video_upload"
)

if therapist_file is not None:
    with open(THERAPIST_VIDEO_PATH, "wb") as f:
        f.write(therapist_file.read())
    duration = get_video_duration(THERAPIST_VIDEO_PATH)
    st.success(f"✅ Therapist video loaded ({duration:.2f} sec)")
    st.video(THERAPIST_VIDEO_PATH)
    st.session_state["therapist_saved"] = True
    st.session_state["therapist_duration"] = duration
else:
    st.session_state["therapist_saved"] = False

# =====================================================
# STEP 2 — PATIENT INPUT (Upload or Record)
# =====================================================

if st.session_state["therapist_saved"]:

    st.divider()
    st.header("Step 2: Patient Session Input")

    input_mode = st.radio(
        "Choose patient input mode", ["Upload Video", "Record Video"],
        horizontal=True, key="patient_input_mode"
    )

    if input_mode == "Upload Video":
        patient_file = st.file_uploader(
            "Upload patient session video",
            type=["mp4", "mov", "avi", "mkv"], key="patient_video_upload"
        )
        if patient_file is not None:
            with open(PATIENT_VIDEO_PATH, "wb") as f:
                f.write(patient_file.read())
            st.success("✅ Patient video uploaded")
            st.session_state["patient_saved"] = True

    else:  # Record Video
        ref_duration = st.session_state.get("therapist_duration", 10.0)
        st.info(f"Recording duration will match therapist video: {ref_duration:.2f} sec")
        if st.button("🎥 Start Recording", key="patient_record_btn"):
            record_video(PATIENT_VIDEO_PATH, ref_duration)
            st.session_state["patient_saved"] = True

    # ── PREVIEW ──────────────────────────────────────

    if st.session_state["patient_saved"] and os.path.exists(PATIENT_VIDEO_PATH):
        st.divider()
        st.subheader("📽 Video Preview")
        prev_col1, prev_col2 = st.columns(2)

        with prev_col1:
            st.markdown("**Therapist Reference**")
            st.video(THERAPIST_VIDEO_PATH)

        with prev_col2:
            st.markdown("**Patient Session**")
            st.video(PATIENT_VIDEO_PATH)

# =====================================================
# STEP 3 — RUN A-SASS ANALYSIS
# =====================================================

if st.session_state["therapist_saved"] and st.session_state["patient_saved"]:

    st.divider()
    st.header("Step 3: Run A-SASS Analysis")
    st.markdown(
        "Compare the **therapist reference** against the **patient session** "
        "using speech, phoneme, and facial articulation analysis."
    )

    if st.button("🔬 Run A-SASS Analysis", use_container_width=True, key="sass_run_btn"):

        with st.spinner("Running Analysis..."):

            # Lazy imports to keep startup fast
            from face_features   import extract_features, compare_features
            from phoneme_utils   import analyze_phonemes
            from attention_rss   import compute_sass
            from session_manager import save_session
            from dashboard       import show_dashboard, attention_chart, articulation_radar

            whisper_model = load_whisper(selected_model)

            therapist_audio = os.path.join(TEMP_DIR, "therapist.wav")
            patient_audio   = os.path.join(TEMP_DIR, "patient.wav")

            extract_audio(THERAPIST_VIDEO_PATH, therapist_audio)
            extract_audio(PATIENT_VIDEO_PATH, patient_audio)

            ref_result = transcribe(whisper_model, therapist_audio, language, TEMP_DIR)
            hyp_result = transcribe(whisper_model, patient_audio, language, TEMP_DIR)

            therapist_text = ref_result["text"]
            patient_text   = hyp_result["text"]

            wer_score = safe_wer(therapist_text, patient_text)

            phoneme_results = analyze_phonemes(therapist_text, patient_text, language)
            per_score       = phoneme_results["per"]

            therapist_features = extract_features(THERAPIST_VIDEO_PATH)
            patient_features   = extract_features(PATIENT_VIDEO_PATH)
            scores             = compare_features(therapist_features, patient_features)

            sass_result = compute_sass(
                wer_score, per_score,
                scores["lip"], scores["mar"], scores["jaw"],
                scores["velocity"], scores["articulation"]
            )

            if patient_name.strip():
                save_session(
                    patient_name, wer_score, per_score,
                    scores["lip"], scores["mar"], scores["jaw"],
                    scores["velocity"], scores["articulation"],
                    sass_result["sass"]
                )

            st.session_state["results"] = {
                "therapist_text":  therapist_text,
                "patient_text":    patient_text,
                "wer_score":       wer_score,
                "per_score":       per_score,
                "scores":          scores,
                "sass_result":     sass_result,
                "phoneme_results": phoneme_results,
                "ref_runs":        ref_result["raw_runs"],
                "hyp_runs":        hyp_result["raw_runs"],
                "ref_romanized":   ref_result["romanized"],
                "hyp_romanized":   hyp_result["romanized"],
            }

    # ── RESULTS DISPLAY ──────────────────────────────

    if st.session_state["results"] is not None:

        R              = st.session_state["results"]
        therapist_text = R["therapist_text"]
        patient_text   = R["patient_text"]
        wer_score      = R["wer_score"]
        per_score      = R["per_score"]
        scores         = R["scores"]
        sass_result    = R["sass_result"]
        phoneme_results = R["phoneme_results"]

        st.divider()
        st.header("📊 Analysis Results")

        if R.get("ref_romanized"):
            st.warning("⚠️ Therapist transcription may be romanized. Try large-v2 model.")
        if R.get("hyp_romanized"):
            st.warning("⚠️ Patient transcription may be romanized. Try large-v2 model.")
        if not therapist_text and not patient_text:
            st.error("❌ No speech detected in either video.")
        elif not patient_text:
            st.warning("⚠️ No speech detected in patient session video.")

        if language == "Hindi":
            with st.expander("🔍 Transcription Debug — 3 voting runs"):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("**Therapist runs:**")
                    for i, r in enumerate(R.get("ref_runs", []), 1):
                        st.markdown(f"Run {i}: `{r}`")
                with dc2:
                    st.markdown("**Patient runs:**")
                    for i, r in enumerate(R.get("hyp_runs", []), 1):
                        st.markdown(f"Run {i}: `{r}`")

        st.subheader("🔤 Transcription — Word Comparison")
        st.markdown(diff_legend_html(), unsafe_allow_html=True)
        ref_html, hyp_html = word_diff_html(therapist_text, patient_text)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Therapist Reference**")
            if therapist_text:
                st.markdown(ref_html, unsafe_allow_html=True)
            else:
                st.warning("_(no speech detected)_")
        with dc2:
            st.markdown("**Real Patient Session**")
            if patient_text:
                st.markdown(hyp_html, unsafe_allow_html=True)
            else:
                st.warning("_(no speech detected)_")

        with st.expander("📄 Raw Transcripts"):
            st.markdown(f"**Therapist:** {therapist_text}")
            st.markdown(f"**Patient:**   {patient_text}")

        st.subheader("Key Scores")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WER",       f"{wer_score:.3f}")
        c2.metric("PER",       f"{per_score:.3f}")
        c3.metric("Lip Score", f"{scores['lip']:.3f}")
        c4.metric("A-SASS",    f"{sass_result['sass'] * 100:.2f}%")

        st.subheader("Visual Articulation Scores")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("MAR",          f"{scores['mar']:.3f}")
        a2.metric("Jaw",          f"{scores['jaw']:.3f}")
        a3.metric("Velocity",     f"{scores['velocity']:.3f}")
        a4.metric("Articulation", f"{scores['articulation']:.3f}")

        st.divider()
        attention_chart(sass_result["weights"])
        articulation_radar(
            scores["lip"], scores["mar"], scores["jaw"],
            scores["velocity"], scores["articulation"]
        )

        st.divider()
        st.subheader("🔤 Phoneme Feedback")
        for item in phoneme_results["feedback"]:
            st.write("•", item)

        st.divider()
        st.subheader("💡 Therapy Recommendation")
        score = sass_result["sass"] * 100
        if score >= 90:
            st.success("✅ Excellent articulation. Patient closely matches the reference.")
        elif score >= 80:
            st.success("✅ Good performance. Minor improvements needed.")
        elif score >= 70:
            st.warning("⚠️ Moderate articulation quality. Continued practice recommended.")
        elif score >= 60:
            st.warning("⚠️ Needs additional practice. Focus on highlighted words.")
        else:
            st.error("❌ Significant articulation issues detected. Intensive therapy advised.")

        if patient_name.strip():
            st.divider()
            st.header(f"📈 Patient Dashboard — {patient_name}")
            show_dashboard(patient_name)
