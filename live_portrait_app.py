# =====================================================
# APP.PY
# LivePortrait + A-SASS Speech Therapy
# =====================================================

import streamlit as st
@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    print("=" * 60)
    print("Loading Whisper model on GPU:", model_name)

    # Check if CUDA is available, otherwise fallback gracefully
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    # CRITICAL: Clear the GPU cache immediately after loading to free up space for LivePortrait
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Finished loading Whisper")
    print("=" * 60)
    return model

try:
    import librosa
    print("librosa imported successfully:", librosa.__version__)
except Exception as e:
    print("librosa import failed:", repr(e))

print("=" * 80)
import tempfile
import subprocess
import os
import cv2

import librosa
import soundfile as sf
import whisper
import numpy as np
import difflib
import shutil

from pathlib import Path
from jiwer import wer

try:
    import mediapipe as _mp_diag
    print("MEDIAPIPE_DIAG file:", getattr(_mp_diag, '__file__', None))
    print("MEDIAPIPE_DIAG version:", getattr(_mp_diag, '__version__', None))
except Exception as _e:
    print("MEDIAPIPE_DIAG import error:", repr(_e))\

# from face_features          import extract_features, compare_features
# from phoneme_utils          import analyze_phonemes
# from attention_rss          import compute_sass
# from session_manager        import save_session, total_sessions, improvement
# from dashboard              import show_dashboard, attention_chart, articulation_radar


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Speech Therapy Suite", page_icon="🧠", layout="wide")
st.title("🧠 Speech Therapy & Reenactment Suite")
st.markdown(
    "Upload a therapist driving input and a patient image to generate "
    "a reenacted reference, then compare it with the patient's real session."
)

# =====================================================
# SESSION STATE INIT
# =====================================================

def _init_state():
    defaults = {
        "driving_saved":    False,
        "patient_saved":    False,
        "reenacted_done":   False,
        "reenacted_format": None,     # "video" or "image"
        "real_saved":       False,
        "results":          None,
        "driving_path":     None,
        "driving_type":     None,     # "video" or "image"
        "_last_driving_type": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# =====================================================
# WHISPER MODEL
# =====================================================

WHISPER_MODEL_OPTIONS = {
    "medium (recommended for Hindi)": "medium",
    "small (faster, less accurate)": "small",
    "large-v2 (best accuracy, slow)": "large-v2",
}

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    print("=" * 60)
    print("Loading Whisper model on CPU to protect LivePortrait GPU VRAM...")
    
    # Force Whisper to use CPU so it won't crash your GPU pipeline
    model = whisper.load_model(model_name, device="cpu")
    
    print("Finished load_whisper")
    print("=" * 60)
    return model
# =====================================================
# TEMP PATHS
# =====================================================

TEMP_DIR             = tempfile.gettempdir()
THERAPIST_VIDEO_PATH = os.path.join(TEMP_DIR, "therapist_video.mp4")
THERAPIST_IMAGE_PATH = os.path.join(TEMP_DIR, "therapist_image.jpg")
PATIENT_IMAGE_PATH   = os.path.join(TEMP_DIR, "patient_image.jpg")
REENACTED_VIDEO_PATH = os.path.join(TEMP_DIR, "reenacted_video.mp4")
REENACTED_IMAGE_PATH = os.path.join(TEMP_DIR, "reenacted_image.png")
REAL_SESSION_PATH    = os.path.join(TEMP_DIR, "real_session_video.mp4")

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
    """
    Transcribes the given audio file using the loaded Whisper model.
    Structures the output to match the application's metric and debug UI.
    """
    if not os.path.exists(audio_path):
        return {"text": "", "raw_runs": [], "romanized": False}

    # Whisper map: 'hi' for Hindi, 'en' for English
    lang_code = "hi" if lang.lower() == "hindi" else "en"
    
    # Run the model transcription
    result = model.transcribe(audio_path, language=lang_code)
    text_out = result.get("text", "").strip()

    # Detect if Hindi transcription accidentally fell back to Latin script (romanized text)
    is_romanized = False
    if lang_code == "hi" and text_out:
        # If more than 80% of characters are standard ASCII, it's romanized
        ascii_count = sum(1 for c in text_out if ord(c) < 128)
        if ascii_count / len(text_out) > 0.8:
            is_romanized = True

    return {
        "text": text_out,
        "raw_runs": [text_out],  # Populates the 3-run debug expander layout gracefully
        "romanized": is_romanized
    }

# =====================================================
# PATIENT INFO + MODEL
# =====================================================

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

# print("3. Before load_whisper")

# with st.spinner(f"Loading Whisper '{selected_model}' model..."):
#     whisper_model = load_whisper(selected_model)

# print("4. After load_whisper")

# st.success("✅ Whisper model loaded successfully!")

if language == "Hindi" and selected_model == "base":
    st.error(
        "⛔ 'base' model romanizes Hindi. "
        "Please select **medium** or **large-v2**."
    )
# =====================================================
# STEP 1 — INPUTS
# =====================================================

st.divider()
st.header("Step 1: Upload Inputs")

# ── DRIVING INPUT ─────────────────────────────────────

st.subheader("🎬 Therapist Driving Input")

driving_type = st.radio(
    "Choose driving input type",
    ["🎥 Video", "🖼️ Image"],
    horizontal=True,
    key="driving_type_selector",
    help=(
        "Video: animates patient face frame-by-frame following therapist motion. "
        "Output will be a reenacted video.\n\n"
        "Image: transfers therapist pose/expression from a single photo. "
        "Output will be a reenacted image."
    )
)

# Reset driving state when type is switched
if st.session_state["_last_driving_type"] != driving_type:
    st.session_state["driving_saved"]    = False
    st.session_state["driving_path"]     = None
    st.session_state["driving_type"]     = None
    st.session_state["reenacted_done"]   = False
    st.session_state["reenacted_format"] = None
    st.session_state["real_saved"]       = False
    st.session_state["results"]          = None
    st.session_state["_last_driving_type"] = driving_type

if driving_type == "🎥 Video":

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
        st.session_state["driving_saved"] = True
        st.session_state["driving_path"]  = THERAPIST_VIDEO_PATH
        st.session_state["driving_type"]  = "video"

else:
    img_driving_opt = st.radio(
        "Choose therapist image source", ["Upload Image", "Capture Image"],
        horizontal=True, key="therapist_image_option"
    )
    therapist_image_file = None
    if img_driving_opt == "Upload Image":
        therapist_image_file = st.file_uploader(
            "Upload therapist image", type=["jpg", "jpeg", "png"],
            key="therapist_image_upload"
        )
    else:
        therapist_image_file = st.camera_input("Capture therapist image", key="therapist_image_camera")

    if therapist_image_file is not None:
        with open(THERAPIST_IMAGE_PATH, "wb") as f:
            f.write(therapist_image_file.getbuffer())
        st.success("✅ Therapist image loaded")
        st.image(therapist_image_file, use_container_width=True)
        st.info("ℹ️ Image driving will produce a reenacted **image** — the therapist's pose transferred onto the patient's face.")
        st.session_state["driving_saved"] = True
        st.session_state["driving_path"]  = THERAPIST_IMAGE_PATH
        st.session_state["driving_type"]  = "image"

st.divider()

# ── PATIENT IMAGE ─────────────────────────────────────

st.subheader("📸 Patient Image (Source Face)")

patient_img_opt = st.radio(
    "Choose patient image source", ["Upload Image", "Capture Image"],
    horizontal=True, key="patient_image_option"
)
patient_image_file = None
if patient_img_opt == "Upload Image":
    patient_image_file = st.file_uploader(
        "Upload patient image", type=["jpg", "jpeg", "png"],
        key="patient_image_upload"
    )
else:
    patient_image_file = st.camera_input("Capture patient image", key="patient_image_camera")

if patient_image_file is not None:
    with open(PATIENT_IMAGE_PATH, "wb") as f:
        f.write(patient_image_file.getbuffer())
    st.success("✅ Patient image loaded")
    st.image(patient_image_file, use_container_width=True)
    st.session_state["patient_saved"] = True

# =====================================================
# STEP 2 — GENERATE REENACTED OUTPUT
# =====================================================

if st.session_state["driving_saved"] and st.session_state["patient_saved"]:

    st.divider()

    is_image_mode = st.session_state["driving_type"] == "image"

    st.header("Step 2: Generate Reenacted Patient " + ("Image" if is_image_mode else "Video"))

    if is_image_mode:
        st.markdown(
            "This will transfer the **therapist's pose/expression** "
            "from the driving image onto the **patient's face** — producing a reenacted **image**."
        )
    else:
        st.markdown(
            "This Model will use the **patient's image** as the source face "
            "and the **therapist's video** as the driving motion — producing a reenacted **video**."
        )

    if st.button("🚀 Generate Reenacted " + ("Image" if is_image_mode else "Video"),
                 use_container_width=True, key="lp_generate_btn"):

        progress = st.progress(0)
        with st.spinner("Running model..."):

            lp_output_dir = os.path.join(TEMP_DIR, "lp_output")
            if os.path.exists(lp_output_dir):
                shutil.rmtree(lp_output_dir)
            os.makedirs(lp_output_dir)
            progress.progress(20)

            command = [
                "python", "inference.py",
                "--source",     PATIENT_IMAGE_PATH,
                "--driving",    st.session_state["driving_path"],
                "--output-dir", lp_output_dir
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                progress.progress(80)

                video_files = list(Path(lp_output_dir).rglob("*.mp4"))
                image_files = (
                    list(Path(lp_output_dir).rglob("*.png")) +
                    list(Path(lp_output_dir).rglob("*.jpg")) +
                    list(Path(lp_output_dir).rglob("*.jpeg"))
                )

                if is_image_mode:
                    # ── IMAGE DRIVING OUTPUT ──────────────────
                    if image_files:
                        shutil.copy(str(image_files[0]), REENACTED_IMAGE_PATH)
                        st.session_state["reenacted_format"] = "image"

                    elif video_files:
                        # LP output mp4 even for image driving — extract first frame
                        subprocess.run([
                            "ffmpeg", "-i", str(video_files[0]),
                            "-frames:v", "1",
                            REENACTED_IMAGE_PATH, "-y"
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        st.session_state["reenacted_format"] = "image"

                    else:
                        raise FileNotFoundError("No output file found")

                else:
                    # ── VIDEO DRIVING OUTPUT ──────────────────
                    if video_files:
                        shutil.copy(str(video_files[0]), REENACTED_VIDEO_PATH)
                        st.session_state["reenacted_format"] = "video"

                    else:
                        raise FileNotFoundError("No output video found")

                progress.progress(100)
                st.session_state["reenacted_done"] = True
                st.session_state["real_saved"]     = False
                st.session_state["results"]        = None
                st.success("✅ Reenacted " + ("image" if is_image_mode else "video") + " generated successfully!")

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                err_msg = e.stderr if hasattr(e, 'stderr') else str(e)
                out_msg = e.stdout if hasattr(e, 'stdout') else result.stdout if 'result' in dir() else ""
                st.error("LivePortrait failed or produced no output.")
                with st.expander("📜 Debug info"):
                    st.text("STDERR:\n" + err_msg)
                    st.text("STDOUT:\n" + out_msg)
                    all_files = list(Path(lp_output_dir).rglob("*"))
                    st.text("Files in output dir:\n" + "\n".join(str(f) for f in all_files))
                st.stop()

        with st.expander("📜 LivePortrait Logs"):
            st.code(result.stdout)

    # ── SHOW REENACTED OUTPUT ─────────────────────────

    if st.session_state["reenacted_done"]:

        fmt = st.session_state["reenacted_format"]

        if fmt == "image" and os.path.exists(REENACTED_IMAGE_PATH):
            st.markdown("#### 🖼️ Reenacted Patient Image")
            st.image(REENACTED_IMAGE_PATH, use_container_width=True)
            with open(REENACTED_IMAGE_PATH, "rb") as f:
                st.download_button(
                    "⬇ Download Reenacted Image", f,
                    "reenacted_patient.png", "image/png",
                    key="lp_download_btn"
                )

        elif fmt == "video" and os.path.exists(REENACTED_VIDEO_PATH):
            st.markdown("#### 🎬 Reenacted Patient Video")
            st.video(REENACTED_VIDEO_PATH)
            with open(REENACTED_VIDEO_PATH, "rb") as f:
                st.download_button(
                    "⬇ Download Reenacted Video", f,
                    "reenacted_patient.mp4", "video/mp4",
                    key="lp_download_btn"
                )

# =====================================================
# STEP 3 — REAL PATIENT SESSION
# =====================================================

if st.session_state["reenacted_done"]:

    st.divider()
    st.header("Step 3: Real Patient Session")
    st.markdown(
        "Upload or record the patient's actual therapy session video "
        "to compare against the reenacted reference."
    )

    input_mode = st.radio(
        "Choose patient session input", ["Upload Video", "Record Video"],
        horizontal=True, key="real_session_mode"
    )

    if input_mode == "Upload Video":
        real_session_file = st.file_uploader(
            "Upload real patient session video",
            type=["mp4", "mov", "avi", "mkv"], key="real_session_upload"
        )
        if real_session_file is not None:
            with open(REAL_SESSION_PATH, "wb") as f:
                f.write(real_session_file.read())
            st.success("✅ Patient session video uploaded")
            st.session_state["real_saved"] = True
    else:
        ref_duration = (
            get_video_duration(REENACTED_VIDEO_PATH)
            if st.session_state["reenacted_format"] == "video"
            else 10.0
        )
        st.info(f"Recording duration: {ref_duration:.2f} sec")
        if st.button("🎥 Start Recording", key="real_session_record_btn"):
            record_video(REAL_SESSION_PATH, ref_duration)
            st.session_state["real_saved"] = True

    if st.session_state["real_saved"] and os.path.exists(REAL_SESSION_PATH):

        st.divider()
        st.subheader("📽 Comparison Preview")
        prev_col1, prev_col2 = st.columns(2)

        with prev_col1:
            st.markdown("**Reenacted Reference**")
            fmt = st.session_state["reenacted_format"]
            if fmt == "image" and os.path.exists(REENACTED_IMAGE_PATH):
                st.image(REENACTED_IMAGE_PATH, use_container_width=True)
            elif fmt == "video" and os.path.exists(REENACTED_VIDEO_PATH):
                st.video(REENACTED_VIDEO_PATH)

        with prev_col2:
            st.markdown("**Real Patient Session**")
            st.video(REAL_SESSION_PATH)

# =====================================================
# STEP 4 — A-SASS ANALYSIS
# =====================================================

if st.session_state["reenacted_done"] and st.session_state["real_saved"]:

    st.divider()
    st.header("Step 4: Run A-SASS Analysis")
    st.markdown(
        "Compare the **reenacted reference** against the **real patient session** "
        "using speech, phoneme, and facial articulation analysis."
    )

    if language == "Hindi" and selected_model == "base":
        st.error("⛔ Switch to **medium** or **large-v2** Whisper model before running Hindi analysis.")
        st.stop()

    # For image-mode reenactment, audio comes from
    # the REAL session only — ref audio uses therapist driving image
    # which has no audio, so we note this to the user
    if st.session_state["reenacted_format"] == "image":
        st.info(
            "ℹ️ The reenacted reference is a static image (no audio). "
            "WER and PER will compare therapist driving audio (if any) "
            "vs patient session audio. Facial scores compare the reenacted "
            "image frame against patient video frames."
        )

    if st.button("🔬 Run A-SASS Analysis", use_container_width=True, key="sass_run_btn"):
       with st.spinner("Running Analysis..."):
            
            # Move your backend imports here to keep app startup 100% stable
            from face_features          import extract_features, compare_features
            from phoneme_utils          import analyze_phonemes
            from attention_rss          import compute_sass
            from session_manager        import save_session, total_sessions, improvement
            from dashboard              import show_dashboard, attention_chart, articulation_radar
            
            whisper_model = load_whisper(selected_model)
            reenacted_audio = os.path.join(TEMP_DIR, "reenacted.wav")
            real_audio      = os.path.join(TEMP_DIR, "real_session.wav")

            fmt = st.session_state["reenacted_format"]

            # For image mode — extract audio from therapist driving image source
            # (no audio in image) → use therapist video if available, else skip
            if fmt == "video":
                extract_audio(REENACTED_VIDEO_PATH, reenacted_audio)
            else:
                # No audio in a static image — create silence placeholder
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "1", reenacted_audio, "-y"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            extract_audio(REAL_SESSION_PATH, real_audio)

            ref_result = transcribe(whisper_model, reenacted_audio, language, TEMP_DIR)
            hyp_result = transcribe(whisper_model, real_audio, language, TEMP_DIR)

            reenacted_text = ref_result["text"]
            real_text      = hyp_result["text"]

            wer_score = safe_wer(reenacted_text, real_text)

            phoneme_results = analyze_phonemes(reenacted_text, real_text, language)
            per_score       = phoneme_results["per"]

            # Face features
            if fmt == "video":
                reenacted_features = extract_features(REENACTED_VIDEO_PATH)
            else:
                # For image mode — extract features from the reenacted image
                reenacted_features = extract_features(REENACTED_IMAGE_PATH)

            real_features = extract_features(REAL_SESSION_PATH)
            scores        = compare_features(reenacted_features, real_features)

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
                "reenacted_text":  reenacted_text,
                "real_text":       real_text,
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

    # ── RESULTS ──────────────────────────────────────

    if st.session_state["results"] is not None:

        R               = st.session_state["results"]
        reenacted_text  = R["reenacted_text"]
        real_text       = R["real_text"]
        wer_score       = R["wer_score"]
        per_score       = R["per_score"]
        scores          = R["scores"]
        sass_result     = R["sass_result"]
        phoneme_results = R["phoneme_results"]

        st.divider()
        st.header("📊 Analysis Results")

        if R.get("ref_romanized"):
            st.warning("⚠️ Reference transcription may be romanized. Try large-v2 model.")
        if R.get("hyp_romanized"):
            st.warning("⚠️ Patient transcription may be romanized. Try large-v2 model.")
        if not reenacted_text and not real_text:
            st.error("❌ No speech detected in either video.")
        elif not real_text:
            st.warning("⚠️ No speech detected in patient session video.")

        if language == "Hindi":
            with st.expander("🔍 Transcription Debug — 3 voting runs"):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("**Reference runs:**")
                    for i, r in enumerate(R.get("ref_runs", []), 1):
                        st.markdown(f"Run {i}: `{r}`")
                with dc2:
                    st.markdown("**Patient runs:**")
                    for i, r in enumerate(R.get("hyp_runs", []), 1):
                        st.markdown(f"Run {i}: `{r}`")

        st.subheader("🔤 Transcription — Word Comparison")
        st.markdown(diff_legend_html(), unsafe_allow_html=True)
        ref_html, hyp_html = word_diff_html(reenacted_text, real_text)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Reenacted Reference**")
            if reenacted_text:
                st.markdown(ref_html, unsafe_allow_html=True)
            else:
                st.warning("_(no speech — image driving has no audio)_")
        with dc2:
            st.markdown("**Real Patient Session**")
            if real_text:
                st.markdown(hyp_html, unsafe_allow_html=True)
            else:
                st.warning("_(no speech detected)_")

        with st.expander("📄 Raw Transcripts"):
            st.markdown(f"**Reference:** {reenacted_text}")
            st.markdown(f"**Patient:**   {real_text}")

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

        # st.subheader("Confusion Matrix")

        # # Safely force the matrix into a clean, display-friendly format
        # import pandas as pd
        # df_matrix = pd.DataFrame(phoneme_results["matrix"])

        # # 2. Stringify the column names and index labels to stop PyArrow from crashing
        # df_matrix.columns = df_matrix.columns.astype(str)
        # df_matrix.index = df_matrix.index.astype(str)

        # # 3. Stringify all values within the cells
        # df_matrix = df_matrix.astype(str)

        # # # 4. Render securely
        # # st.dataframe(df_matrix, width='stretch')
        # try:
        #     st.dataframe(some_variable)
        # except Exception:
        #     # If pyarrow fails, it drops back to st.write which NEVER crashes
        #     st.write(some_variable)
        # st.divider()
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