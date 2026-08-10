# =====================================================
# APP.PY
# PART 1
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

from jiwer import wer

# =====================================================
# CUSTOM MODULES
# =====================================================

from face_features import (
    extract_features,
    compare_features
)

from phoneme_utils import (
    analyze_phonemes
)

from attention_rss import (
    compute_sass
)

from session_manager import (
    save_session,
    total_sessions,
    improvement
)

from dashboard import (
    show_dashboard,
    attention_chart,
    articulation_radar
)

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="A-SASS Speech Therapy",
    page_icon="🧠",
    layout="wide"
)

st.title(
    "🧠 Attention-Based Speech Therapy System"
)

st.markdown(
"""
### Features

✅ Whisper Speech Recognition

✅ Phoneme Error Analysis

✅ Lip Similarity

✅ Mouth Aspect Ratio

✅ Jaw Motion Analysis

✅ Velocity Analysis

✅ Attention-Based Speech Articulation Similarity Score (A-SASS)

✅ Patient Progress Dashboard
"""
)

# =====================================================
# LANGUAGE
# =====================================================

language = st.selectbox(

    "Language",

    [
        "English",
        "Hindi"
    ]

)

# =====================================================
# WHISPER MODEL SELECTION
# =====================================================

model_choice = st.selectbox(
    "Whisper Model",
    [
        "base",
        "medium",
        "large"
    ],
    index=0,
    help="Base = fastest, lower accuracy. Medium = balanced. Large = best accuracy, slower."
)

st.info(
    f"Selected: **{model_choice}** — Loading this will take time on first run."
)

# =====================================================
# PATIENT
# =====================================================

patient_name = st.text_input(
    "Patient Name"
)

# =====================================================
# WHISPER
# =====================================================

@st.cache_resource
def load_whisper(model_size):

    model = whisper.load_model(model_size)

    return model

whisper_model = load_whisper(model_choice)

# =====================================================
# AUDIO EXTRACTION
# =====================================================

def extract_audio(
        video_path,
        audio_path
):

    command = [

        "ffmpeg",

        "-i",
        video_path,

        "-ac",
        "1",

        "-ar",
        "16000",

        "-vn",

        audio_path,

        "-y"

    ]

    subprocess.run(

        command,

        stdout=subprocess.DEVNULL,

        stderr=subprocess.DEVNULL

    )

# =====================================================
# NORMALIZE AUDIO
# =====================================================

def normalize_audio(
        input_audio,
        output_audio
):

    y, sr = librosa.load(

        input_audio,

        sr=16000

    )

    y = librosa.util.normalize(
        y
    )

    sf.write(

        output_audio,

        y,

        sr

    )

# =====================================================
# TRANSCRIPTION
# =====================================================

def transcribe_audio(
        audio_path,
        language
):

    try:

        if language == "English":

            result = whisper_model.transcribe(
                audio_path
            )

        else:

            result = whisper_model.transcribe(

                audio_path,

                language="hi"

            )

        return result["text"].strip()

    except:

        return ""

# =====================================================
# SAFE WER
# =====================================================

def safe_wer(
        ref,
        hyp
):

    ref = ref.strip()
    hyp = hyp.strip()

    if ref == "" and hyp == "":
        return 0

    if ref == "" or hyp == "":
        return 1

    try:

        return wer(
            ref,
            hyp
        )

    except:

        return 1

# =====================================================
# VIDEO DURATION
# =====================================================

def get_video_duration(
        video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frames = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    cap.release()

    if fps == 0:
        return 0

    return frames / fps

# =====================================================
# RECORD VIDEO
# =====================================================

def record_video(
        output_path,
        duration
):

    cap = cv2.VideoCapture(0)

    width = 640
    height = 480

    writer = cv2.VideoWriter(

        output_path,

        cv2.VideoWriter_fourcc(*'mp4v'),

        20,

        (width,height)

    )

    st.info(
        f"Recording {duration:.2f} seconds..."
    )

    total_frames = int(
        duration * 20
    )

    for _ in range(total_frames):

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

    cap.release()

    writer.release()

    st.success(
        "Recording Completed"
    )
    # =====================================================
# PART 2
# VIDEO INPUT
# =====================================================

st.divider()

st.header(
    "Step 1: Upload Therapist Video"
)

therapist_file = st.file_uploader(

    "Therapist Video",

    type=["mp4"]

)

# =====================================================
# TEMP FILE PATHS
# =====================================================

therapist_video_path = os.path.join(

    tempfile.gettempdir(),

    "therapist_video.mp4"

)

patient_video_path = os.path.join(

    tempfile.gettempdir(),

    "patient_video.mp4"

)

# =====================================================
# SAVE THERAPIST VIDEO
# =====================================================

if therapist_file:

    with open(
        therapist_video_path,
        "wb"
    ) as f:

        f.write(
            therapist_file.read()
        )

    duration = get_video_duration(
        therapist_video_path
    )

    st.success(
        f"Therapist video loaded ({duration:.2f} sec)"
    )

    # ===============================================
    # STEP 2
    # ===============================================

    st.divider()

    st.header(
        "Step 2: Patient Input"
    )

    input_mode = st.radio(

        "Choose Input Mode",

        [
            "Upload Video",
            "Record Video"
        ]

    )

    # ===============================================
    # UPLOAD PATIENT VIDEO
    # ===============================================

    if input_mode == "Upload Video":

        patient_file = st.file_uploader(

            "Patient Video",

            type=["mp4"]

        )

        if patient_file:

            with open(
                patient_video_path,
                "wb"
            ) as f:

                f.write(
                    patient_file.read()
                )

            st.success(
                "Patient video uploaded"
            )

    # ===============================================
    # RECORD PATIENT VIDEO
    # ===============================================

    if input_mode == "Record Video":

        st.info(
            "Recording duration will match therapist video."
        )

        if st.button(
            "🎥 Record Patient Video"
        ):

            record_video(

                patient_video_path,

                duration

            )

    # ===============================================
    # VIDEO PREVIEW
    # ===============================================

    st.divider()

    st.header(
        "Video Preview"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Therapist"
        )

        st.video(
            therapist_video_path
        )

    with col2:

        st.subheader(
            "Patient"
        )

        if os.path.exists(
            patient_video_path
        ):

            st.video(
                patient_video_path
            )

        else:

            st.warning(
                "Patient video not available"
            )

    # ===============================================
    # VALIDATION
    # ===============================================

    ready = (
        os.path.exists(
            therapist_video_path
        )
        and
        os.path.exists(
            patient_video_path
        )
    )

    if ready:

        st.success(
            "Videos ready for analysis"
        )

    else:

        st.warning(
            "Upload or record patient video"
        )

    # ===============================================
    # ANALYSIS BUTTON
    # ===============================================

    if ready:

        run_analysis = st.button(

            "🚀 Run Analysis",

            use_container_width=True

        )
        if run_analysis:

            with st.spinner(
                "Running Analysis..."
            ):

                # ==========================
                # AUDIO
                # ==========================

                therapist_audio = "therapist.wav"
                patient_audio = "patient.wav"

                extract_audio(
                    therapist_video_path,
                    therapist_audio
                )

                extract_audio(
                    patient_video_path,
                    patient_audio
                )

                normalize_audio(
                    therapist_audio,
                    therapist_audio
                )

                normalize_audio(
                    patient_audio,
                    patient_audio
                )

                # ==========================
                # TRANSCRIPTION
                # ==========================

                therapist_text = transcribe_audio(
                    therapist_audio,
                    language
                )

                patient_text = transcribe_audio(
                    patient_audio,
                    language
                )

                # ==========================
                # WER
                # ==========================

                wer_score = safe_wer(
                    therapist_text,
                    patient_text
                )

                # ==========================
                # PER
                # ==========================

                phoneme_results = analyze_phonemes(
                    therapist_text,
                    patient_text,
                    language
                )

                per_score = phoneme_results["per"]

                # ==========================
                # FACE FEATURES
                # ==========================

                therapist_features = extract_features(
                    therapist_video_path
                )

                patient_features = extract_features(
                    patient_video_path
                )

                scores = compare_features(
                    therapist_features,
                    patient_features
                )

                # ==========================
                # A-SASS
                # ==========================

                sass_result = compute_sass(

                    wer_score,

                    per_score,

                    scores["lip"],

                    scores["mar"],

                    scores["jaw"],

                    scores["velocity"],

                    scores["articulation"]

                )

                # ==========================
                # SAVE SESSION
                # ==========================

                if patient_name.strip():

                    save_session(

                        patient_name,

                        wer_score,

                        per_score,

                        scores["lip"],

                        scores["mar"],

                        scores["jaw"],

                        scores["velocity"],

                        scores["articulation"],

                        sass_result["sass"]

                    )

                st.divider()

            st.header(
                "Analysis Results"
            )

            st.subheader(
                "Transcriptions"
            )

            st.write(
                "Therapist:",
                therapist_text
            )

            st.write(
                "Patient:",
                patient_text
            )

            c1,c2,c3,c4 = st.columns(4)

            c1.metric(
                "WER",
                f"{wer_score:.3f}"
            )

            c2.metric(
                "PER",
                f"{per_score:.3f}"
            )

            c3.metric(
                "Lip Score",
                f"{scores['lip']:.3f}"
            )

            c4.metric(
                "A-SASS",
                f"{sass_result['sass']*100:.2f}%"
            )

            st.subheader(
                " Visual Articulation Scores"
            )

            col1,col2,col3,col4 = st.columns(4)

            col1.metric(
                "MAR",
                f"{scores['mar']:.3f}"
            )

            col2.metric(
                "Jaw",
                f"{scores['jaw']:.3f}"
            )

            col3.metric(
                "Velocity",
                f"{scores['velocity']:.3f}"
            )

            col4.metric(
                " Visual Articulation",
                f"{scores['articulation']:.3f}"
            )

            st.divider()

            attention_chart(
                sass_result["weights"]
            )

            articulation_radar(

                scores["lip"],

                scores["mar"],

                scores["jaw"],

                scores["velocity"],

                scores["articulation"]

            )

            st.divider()

            st.subheader(
                "Phoneme Feedback"
            )

            for item in phoneme_results[
                "feedback"
            ]:

                st.write(
                    "•",
                    item
                )

            st.subheader(
                "Confusion Matrix"
            )

            st.dataframe(
                phoneme_results["matrix"]
            )

            st.divider()

            st.subheader(
                "Therapy Recommendation"
            )

            score = (
                sass_result["sass"] * 100
            )

            if score >= 90:

                st.success(
                    "Excellent articulation."
                )

            elif score >= 80:

                st.success(
                    "Good performance."
                )

            elif score >= 70:

                st.warning(
                    "Moderate articulation quality, little practice required."
                )

            elif score >= 60:

                st.warning(
                    "Needs additional practice."
                )

            else:

                st.error(
                    "Significant articulation issues detected."
                )

            if patient_name.strip():

                st.divider()

                st.header(
                    "Patient Dashboard"
                )

                show_dashboard(
                    patient_name
                )