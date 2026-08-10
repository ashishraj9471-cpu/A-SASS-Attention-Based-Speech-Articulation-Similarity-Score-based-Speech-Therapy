import pandas as pd
import os
from datetime import datetime

# =====================================================
# DEFAULT FILE
# =====================================================

CSV_FILE = "therapy_sessions.csv"

# =====================================================
# CREATE CSV
# =====================================================

def initialize_database():

    if not os.path.exists(CSV_FILE):

        df = pd.DataFrame(columns=[

            "Timestamp",

            "Patient",

            "WER",

            "PER",

            "Lip",

            "MAR",

            "Jaw",

            "Velocity",

            "Articulation",

            "A_SASS"

        ])

        df.to_csv(
            CSV_FILE,
            index=False
        )

# =====================================================
# SAVE SESSION
# =====================================================

def save_session(

        patient_name,

        wer_score,

        per_score,

        lip_score,

        mar_score,

        jaw_score,

        velocity_score,

        articulation_score,

        sass_score

):

    initialize_database()

    row = {

        "Timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Patient":
            patient_name,

        "WER":
            wer_score,

        "PER":
            per_score,

        "Lip":
            lip_score,

        "MAR":
            mar_score,

        "Jaw":
            jaw_score,

        "Velocity":
            velocity_score,

        "Articulation":
            articulation_score,

        "A_SASS":
            sass_score

    }

    df = pd.read_csv(
        CSV_FILE
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame([row])
        ],
        ignore_index=True
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )

# =====================================================
# LOAD ALL
# =====================================================

def load_sessions():

    initialize_database()

    return pd.read_csv(
        CSV_FILE
    )

# =====================================================
# PATIENT HISTORY
# =====================================================

def patient_history(
        patient_name
):

    df = load_sessions()

    df = df[
        df["Patient"] == patient_name
    ]

    return df

# =====================================================
# LAST SESSION
# =====================================================

def last_session(
        patient_name
):

    history = patient_history(
        patient_name
    )

    if len(history) == 0:

        return None

    return history.iloc[-1]

# =====================================================
# SESSION COUNT
# =====================================================

def total_sessions(
        patient_name
):

    history = patient_history(
        patient_name
    )

    return len(history)

# =====================================================
# IMPROVEMENT
# =====================================================

def improvement(
        patient_name
):

    history = patient_history(
        patient_name
    )

    if len(history) < 2:

        return None

    first = history.iloc[0]
    last = history.iloc[-1]

    return {

        "WER":

        first["WER"]
        -
        last["WER"],

        "PER":

        first["PER"]
        -
        last["PER"],

        "A_SASS":

        last["A_SASS"]
        -
        first["A_SASS"]

    }

# =====================================================
# BEST SESSION
# =====================================================

def best_session(
        patient_name
):

    history = patient_history(
        patient_name
    )

    if len(history) == 0:

        return None

    idx = history[
        "A_SASS"
    ].idxmax()

    return history.loc[idx]

# =====================================================
# SUMMARY
# =====================================================

def patient_summary(
        patient_name
):

    history = patient_history(
        patient_name
    )

    if len(history) == 0:

        return None

    return {

        "Sessions":

        len(history),

        "Average_WER":

        history["WER"].mean(),

        "Average_PER":

        history["PER"].mean(),

        "Average_A_SASS":

        history["A_SASS"].mean(),

        "Best_A_SASS":

        history["A_SASS"].max()

    }

# =====================================================
# EXPORT
# =====================================================

def export_patient(
        patient_name
):

    history = patient_history(
        patient_name
    )

    filename = (
        patient_name +
        "_history.csv"
    )

    history.to_csv(
        filename,
        index=False
    )

    return filename