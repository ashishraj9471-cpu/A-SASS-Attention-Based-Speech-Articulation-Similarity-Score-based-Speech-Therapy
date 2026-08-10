# =====================================================
# ATTENTION_RSS.PY
# Attention-Based Speech Articulation Similarity Score
#
# Hindi speech therapy context:
#
#   PROBLEM with pure softmax over feature values:
#     → WER gets high attention only when patient is
#       performing badly (low similarity = high softmax).
#       When patient improves, WER weight drops — which
#       is clinically wrong. WER must always dominate.
#
#   SOLUTION — 3-tier hybrid weighting:
#     Tier 1: WER         = 0.50 (fixed, always dominant)
#     Tier 2: PER         = 0.15  (fixed, reduced for Hindi)
#     Tier 3: facial x5  = 0.35  (softmax over deviations,
#                                  so worst facial feature
#                                  gets most attention)
# =====================================================

import numpy as np
import pandas as pd

# =====================================================
# FIXED TIER WEIGHTS
# =====================================================

W_WER = 0.45  # Tier 1 — always dominant
W_PER = 0.15   # Tier 2 — reduced for Hindi
W_T3  = 0.40  # Tier 3 — split by softmax among facial features

FEATURE_NAMES = [
    "WER",
    "PER",
    "Lip",
    "MAR",
    "Jaw",
    "Velocity",
    "Articulation"
]

# =====================================================
# SOFTMAX (numerically stable)
# =====================================================

def softmax(x):

    x = np.array(x, dtype=np.float32)
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x) + 1e-8)

# =====================================================
# TIER 3 ATTENTION
#
# Softmax is applied over *deviations from perfect*,
# not over raw scores. This way the facial feature
# that needs the most work gets the most attention —
# regardless of whether WER is high or low.
# =====================================================

def tier3_attention(
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
):

    tier3_scores = np.array([
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
    ], dtype=np.float32)

    # Deviation from perfect (1.0) — higher = needs more attention
    deviations = np.abs(1.0 - tier3_scores)

    # Softmax over deviations → attention distribution
    attn = softmax(deviations)

    # Scale to Tier 3 budget
    return attn * W_T3

# =====================================================
# BUILD SIMILARITIES
#
# WER and PER are error scores → convert to similarity.
# Facial scores are already similarity values.
# =====================================================

def build_similarities(
        wer_score,
        per_score,
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
):

    return np.array([
        max(0.0, 1.0 - wer_score),      # WER → similarity
        max(0.0, 1.0 - per_score),      # PER → similarity
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
    ], dtype=np.float32)

# =====================================================
# FINAL WEIGHT VECTOR
# =====================================================

def build_weights(
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
):

    t3_weights = tier3_attention(
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
    )

    return np.array([
        W_WER,
        W_PER,
        t3_weights[0],   # lip
        t3_weights[1],   # mar
        t3_weights[2],   # jaw
        t3_weights[3],   # velocity
        t3_weights[4],   # articulation
    ], dtype=np.float32)

# =====================================================
# PERFORMANCE CATEGORY
# =====================================================

def sass_category(score):

    score = score * 100

    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Moderate"
    elif score >= 60:
        return "Needs Improvement"
    else:
        return "Poor"

# =====================================================
# EXPLAIN ATTENTION
# =====================================================

def explain_attention(weights):

    return {
        name: float(w)
        for name, w in zip(FEATURE_NAMES, weights)
    }

# =====================================================
# ATTENTION DATAFRAME
# =====================================================

def attention_dataframe(weights):

    return pd.DataFrame({
        "Feature":    FEATURE_NAMES,
        "Weight":     [round(float(w), 4) for w in weights],
        "Weight (%)": [round(float(w) * 100, 2) for w in weights]
    })

# =====================================================
# COMPUTE A-SASS  (drop-in replacement)
# =====================================================

def compute_sass(
        wer_score,
        per_score,
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
):

    similarities = build_similarities(
        wer_score,
        per_score,
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
    )

    weights = build_weights(
        lip_score,
        mar_score,
        jaw_score,
        velocity_score,
        articulation_score
    )

    sass = float(np.sum(weights * similarities))

    category     = sass_category(sass)
    explanation  = explain_attention(weights)
    table        = attention_dataframe(weights)

    return {
        "sass":        sass,
        "weights":     weights,        # np.array — for attention_chart()
        "category":    category,
        "table":       table,
        "explanation": explanation,
    }