import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from session_manager import (
    patient_history,
    patient_summary,
    best_session
)

# =====================================================
# OVERVIEW
# =====================================================

def show_summary(patient_name):

    summary = patient_summary(
        patient_name
    )

    if summary is None:

        st.warning(
            "No session history found."
        )

        return

    st.subheader(
        "Patient Summary"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Sessions",
        summary["Sessions"]
    )

    c2.metric(
        "Avg WER",
        round(
            summary["Average_WER"],
            3
        )
    )

    c3.metric(
        "Avg PER",
        round(
            summary["Average_PER"],
            3
        )
    )

    c4.metric(
        "Avg A-SASS",
        round(
            summary["Average_A_SASS"]*100,
            2
        )
    )

# =====================================================
# SESSION TABLE
# =====================================================

def show_history(patient_name):

    history = patient_history(
        patient_name
    )

    if len(history) == 0:

        st.warning(
            "No data available."
        )

        return

    st.subheader(
        "Session History"
    )

    st.dataframe(
        history,
        use_container_width=True
    )

# =====================================================
# A-SASS TREND
# =====================================================

def sass_trend(patient_name):

    history = patient_history(
        patient_name
    )

    if len(history) < 2:

        return

    st.subheader(
        "A-SASS Progress"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.plot(
        history["A_SASS"],
        marker="o"
    )

    ax.set_ylabel(
        "A-SASS"
    )

    ax.set_xlabel(
        "Session"
    )

    st.pyplot(fig)

# =====================================================
# WER TREND
# =====================================================

def wer_trend(patient_name):

    history = patient_history(
        patient_name
    )

    if len(history) < 2:

        return

    st.subheader(
        "WER Progress"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.plot(
        history["WER"],
        marker="o"
    )

    ax.set_ylabel(
        "WER"
    )

    ax.set_xlabel(
        "Session"
    )

    st.pyplot(fig)

# =====================================================
# PER TREND
# =====================================================

def per_trend(patient_name):

    history = patient_history(
        patient_name
    )

    if len(history) < 2:

        return

    st.subheader(
        "PER Progress"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.plot(
        history["PER"],
        marker="o"
    )

    ax.set_ylabel(
        "PER"
    )

    ax.set_xlabel(
        "Session"
    )

    st.pyplot(fig)

# =====================================================
# MULTI METRIC
# =====================================================

def multi_metric_chart(
        patient_name
):

    history = patient_history(
        patient_name
    )

    if len(history) < 2:

        return

    st.subheader(
        "Therapy Progress"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.plot(
        history["A_SASS"],
        label="A-SASS"
    )

    ax.plot(
        history["Lip"],
        label="Lip"
    )

    ax.plot(
        history["MAR"],
        label="MAR"
    )

    ax.plot(
        history["Jaw"],
        label="Jaw"
    )

    ax.plot(
        history["Velocity"],
        label="Velocity"
    )

    ax.legend()

    st.pyplot(fig)

# =====================================================
# BEST SESSION
# =====================================================

def show_best_session(
        patient_name
):

    best = best_session(
        patient_name
    )

    if best is None:

        return

    st.subheader(
        "Best Session"
    )

    st.write(
        best
    )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def attention_chart(
        weights
):

    st.subheader(
        "Attention Weights"
    )

    labels = [

        "WER",
        "PER",
        "Lip",
        "MAR",
        "Jaw",
        "Velocity",
        "Articulation"

    ]

    df = pd.DataFrame({

        "Feature":
        labels,

        "Weight":
        weights

    })

    st.bar_chart(
        df.set_index(
            "Feature"
        )
    )

# =====================================================
# RADAR CHART
# =====================================================

def articulation_radar(

        lip,

        mar,

        jaw,

        velocity,

        articulation

):

    import numpy as np

    categories = [

        "Lip",
        "MAR",
        "Jaw",
        "Velocity",
        "Articulation"

    ]

    values = [

        lip,
        mar,
        jaw,
        velocity,
        articulation

    ]

    values += values[:1]

    angles = np.linspace(
        0,
        2*np.pi,
        len(categories),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(
        figsize=(6,6)
    )

    ax = plt.subplot(
        111,
        polar=True
    )

    ax.plot(
        angles,
        values
    )

    ax.fill(
        angles,
        values,
        alpha=0.2
    )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        categories
    )

    st.pyplot(fig)

# =====================================================
# FULL DASHBOARD
# =====================================================

def show_dashboard(
        patient_name
):

    show_summary(
        patient_name
    )

    show_history(
        patient_name
    )

    sass_trend(
        patient_name
    )

    wer_trend(
        patient_name
    )

    per_trend(
        patient_name
    )

    multi_metric_chart(
        patient_name
    )

    show_best_session(
        patient_name
    )