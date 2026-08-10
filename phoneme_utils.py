# =====================================================
# PHONEME_UTILS.PY — Hindi + English Phoneme Analysis
# =====================================================

import re
import numpy as np
import pandas as pd
from collections import Counter

# =====================================================
# HINDI TEXT NORMALIZATION
# =====================================================

# Nukta variations → base character (standardize)
NUKTA_MAP = {
    'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज', 'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ',
    'य़': 'य',
}

# Common normalization: chandrabindu → anusvara, double matra fix
def normalize_hindi(text: str) -> str:
    if not text:
        return ""
    
    # If romanized, return as-is (phoneme_utils will handle via fallback)
    # Check if mostly ASCII
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio > 0.7:
        return text  # Let romanized path handle it
    
    # Replace nukta characters
    for src, dst in NUKTA_MAP.items():
        text = text.replace(src, dst)
    
    # Normalize chandrabindu (ँ) to anusvara (ं) for simplicity
    text = text.replace('ँ', 'ं')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# =====================================================
# HINDI PHONEME MAPPER (Devanagari → Phoneme classes)
# =====================================================

# Vowels (swar) — standalone and matra forms
HINDI_VOWELS = set('अआइईउऊऋएऐओऔअंअः')
HINDI_MATRAS = {
    'ा': 'आ', 'ि': 'इ', 'ी': 'ई', 'ु': 'उ', 'ू': 'ऊ',
    'ृ': 'ऋ', 'े': 'ए', 'ै': 'ऐ', 'ो': 'ओ', 'ौ': 'औ',
    'ं': 'अं', 'ः': 'अः', '्': ''  # halant = no vowel
}

# Consonants mapped to phoneme symbols (IPA-like simplified)
HINDI_CONSONANT_PHONEMES = {
    'क': 'k', 'ख': 'kʰ', 'ग': 'g', 'घ': 'gʰ', 'ङ': 'ŋ',
    'च': 'c', 'छ': 'cʰ', 'ज': 'j', 'झ': 'jʰ', 'ञ': 'ɲ',
    'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
    'त': 't', 'थ': 'tʰ', 'द': 'd', 'ध': 'dʰ', 'न': 'n',
    'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'ळ': 'ɭ',
    'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'h',
    'क्ष': 'kʂ', 'त्र': 'tr', 'ज्ञ': 'gj', 'श्र': 'ʃr',
}

HINDI_VOWEL_PHONEMES = {
    'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː',
    'उ': 'ʊ', 'ऊ': 'uː', 'ऋ': 'r̩', 'ए': 'eː',
    'ऐ': 'ɛː', 'ओ': 'oː', 'औ': 'ɔː',
    'अं': 'ə̃', 'अः': 'əh',
}

def devanagari_to_phonemes(text: str) -> list:
    """
    Convert Devanagari text to a list of phoneme tokens.
    Handles consonant clusters (halant) and matras.
    """
    text = normalize_hindi(text)
    phonemes = []
    i = 0
    n = len(text)
    
    while i < n:
        ch = text[i]
        
        # Skip whitespace
        if ch.isspace():
            i += 1
            continue
        
        # Check for two-char conjuncts first (क्ष, त्र, ज्ञ, श्र)
        if i + 1 < n:
            two = text[i:i+2]
            if two in HINDI_CONSONANT_PHONEMES:
                phonemes.append(HINDI_CONSONANT_PHONEMES[two])
                i += 2
                # Check for matra after conjunct
                if i < n and text[i] in HINDI_MATRAS:
                    matra = HINDI_MATRAS[text[i]]
                    if matra:
                        phonemes.append(HINDI_VOWEL_PHONEMES.get(matra, matra))
                    i += 1
                elif i < n and text[i] == '्':
                    i += 1  # halant already consumed in logic above? no, two-char has no halant
                continue
        
        # Single consonant
        if ch in HINDI_CONSONANT_PHONEMES:
            phonemes.append(HINDI_CONSONANT_PHONEMES[ch])
            i += 1
            # Check for matra or halant
            if i < n and text[i] in HINDI_MATRAS:
                matra = HINDI_MATRAS[text[i]]
                if matra:
                    phonemes.append(HINDI_VOWEL_PHONEMES.get(matra, matra))
                i += 1
            # Implicit 'ə' if no matra/halant and next is not matra
            # (simplified: assume schwa is present unless halant)
            continue
        
        # Standalone vowel
        if ch in HINDI_VOWEL_PHONEMES:
            phonemes.append(HINDI_VOWEL_PHONEMES[ch])
            i += 1
            continue
        
        # Matra appearing standalone (shouldn't happen but handle)
        if ch in HINDI_MATRAS:
            matra = HINDI_MATRAS[ch]
            if matra:
                phonemes.append(HINDI_VOWEL_PHONEMES.get(matra, matra))
            i += 1
            continue
        
        # Unknown character (punctuation, digit, etc.)
        i += 1
    
    return phonemes

# =====================================================
# ROMANIZED HINDI FALLBACK (Simple character-level)
# =====================================================

def romanized_to_phonemes(text: str) -> list:
    """Fallback for romanized Hindi: treat each character as phoneme."""
    text = re.sub(r'[^a-zA-Z]', '', text.lower())
    return list(text)

# =====================================================
# ENGLISH PHONEMES (CMUdict fallback)
# =====================================================

def english_to_phonemes(text: str) -> list:
    try:
        import cmudict
        arpabet = cmudict.dict()
        words = re.findall(r"[a-zA-Z']+", text.lower())
        phonemes = []
        for w in words:
            if w in arpabet:
                phonemes.extend(arpabet[w][0])
            else:
                phonemes.extend(list(w))  # char fallback
        return phonemes
    except Exception:
        # No cmudict installed
        return list(re.sub(r'[^a-zA-Z]', '', text.lower()))

# =====================================================
# SMART PHONEME EXTRACTOR
# =====================================================

def extract_phonemes(text: str, language: str) -> list:
    if not text or not text.strip():
        return []
    
    # Detect if text is Devanagari or Romanized/English
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = sum(1 for c in text if c.isalpha())
    
    if language == "Hindi" and devanagari_chars > 0:
        return devanagari_to_phonemes(text)
    elif language == "Hindi":
        # Romanized Hindi fallback
        return romanized_to_phonemes(text)
    else:
        return english_to_phonemes(text)

# =====================================================
# LEVENSHTEIN PER
# =====================================================

def levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def compute_per(ref_phonemes: list, hyp_phonemes: list) -> float:
    if len(ref_phonemes) == 0 and len(hyp_phonemes) == 0:
        return 0.0
    if len(ref_phonemes) == 0 or len(hyp_phonemes) == 0:
        return 1.0
    dist = levenshtein(ref_phonemes, hyp_phonemes)
    return min(dist / len(ref_phonemes), 1.0)

# =====================================================
# CONFUSION MATRIX & FEEDBACK
# =====================================================

def phoneme_confusion(ref: list, hyp: list):
    """Build confusion matrix from aligned phonemes."""
    # Simple alignment using Levenshtein traceback
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    
    # Traceback
    i, j = m, n
    aligned_ref, aligned_hyp = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            aligned_ref.append(ref[i-1])
            aligned_hyp.append(hyp[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            aligned_ref.append(ref[i-1])
            aligned_hyp.append('-')
            i -= 1
        else:
            aligned_ref.append('-')
            aligned_hyp.append(hyp[j-1])
            j -= 1
    
    aligned_ref.reverse()
    aligned_hyp.reverse()
    
    # Build confusion matrix
    errors = Counter()
    for r, h in zip(aligned_ref, aligned_hyp):
        if r != h:
            errors[(r, h)] += 1
    
    # Create DataFrame
    if errors:
        df = pd.DataFrame([
            {"Expected": r, "Got": h, "Count": c}
            for (r, h), c in errors.most_common(15)
        ])
    else:
        df = pd.DataFrame(columns=["Expected", "Got", "Count"])
    
    return df, aligned_ref, aligned_hyp

def generate_feedback(ref: list, hyp: list, aligned_ref: list, aligned_hyp: list, language: str) -> list:
    """Generate human-readable feedback from phoneme alignment."""
    feedback = []
    
    if len(ref) == 0:
        return ["No reference phonemes detected."]
    if len(hyp) == 0:
        return ["No patient phonemes detected. Please speak clearly."]
    
    per = compute_per(ref, hyp)
    
    if per == 0:
        return ["Perfect phoneme match!"]
    
    # Count error types
    substitutions = sum(1 for r, h in zip(aligned_ref, aligned_hyp) if r != h and r != '-' and h != '-')
    deletions = sum(1 for r, h in zip(aligned_ref, aligned_hyp) if h == '-' and r != '-')
    insertions = sum(1 for r, h in zip(aligned_ref, aligned_hyp) if r == '-' and h != '-')
    
    feedback.append(f"Phoneme Error Rate: {per*100:.1f}%")
    feedback.append(f"Errors — Substitutions: {substitutions}, Deletions: {deletions}, Insertions: {insertions}")
    
    if language == "Hindi":
        # Hindi-specific feedback
        retroflex = ['ʈ', 'ʈʰ', 'ɖ', 'ɖʰ', 'ɳ', 'ɭ', 'ʂ']
        dental = ['t', 'tʰ', 'd', 'dʰ', 'n', 'l', 's']
        
        retro_errors = []
        dental_errors = []
        asp_errors = []
        
        for r, h in zip(aligned_ref, aligned_hyp):
            if r != h and r != '-' and h != '-':
                if r in retroflex and h in dental:
                    retro_errors.append(f"Used dental '{h}' instead of retroflex '{r}'")
                elif r in dental and h in retroflex:
                    dental_errors.append(f"Used retroflex '{h}' instead of dental '{r}'")
                elif (r.endswith('ʰ') and not h.endswith('ʰ')) or (not r.endswith('ʰ') and h.endswith('ʰ')):
                    asp_errors.append(f"Aspiration mismatch: expected '{r}', got '{h}'")
        
        if retro_errors:
            feedback.append("⚠️ Retroflex confusion detected (ट/त, ड/द, ण/न). Practice tongue placement behind teeth ridge.")
        if dental_errors:
            feedback.append("⚠️ Dental/retroflex mix-up. Focus on curling tongue back for ट, ड, ण.")
        if asp_errors:
            feedback.append("⚠️ Aspirated vs unaspirated confusion (क/ख, ग/घ). Listen for the 'h' breath sound.")
            
        # Vowel length
        long_vowels = ['aː', 'iː', 'uː', 'eː', 'oː', 'ɛː', 'ɔː']
        short_vowels = ['ə', 'ɪ', 'ʊ']
        vowel_errors = []
        for r, h in zip(aligned_ref, aligned_hyp):
            if r != h and r != '-' and h != '-':
                if r in long_vowels and h in short_vowels:
                    vowel_errors.append(f"Shortened long vowel: '{r}' → '{h}'")
                elif r in short_vowels and h in long_vowels:
                    vowel_errors.append(f"Lengthened short vowel: '{r}' → '{h}'")
        if vowel_errors:
            feedback.append("⚠️ Vowel length errors (matra duration). Hold long vowels (आ, ई, ऊ) longer.")
    
    else:
        # English feedback
        if substitutions > 0:
            feedback.append(f"⚠️ {substitutions} phoneme substitutions detected.")
        if deletions > 0:
            feedback.append(f"⚠️ {deletions} phonemes missing (deletions).")
        if insertions > 0:
            feedback.append(f"⚠️ {insertions} extra phonemes inserted.")
    
    return feedback

# =====================================================
# MAIN ANALYZE FUNCTION (API-compatible with old code)
# =====================================================

def analyze_phonemes(ref_text: str, hyp_text: str, language: str) -> dict:
    """
    Analyze phonemes between reference and hypothesis.
    Returns dict with: per, matrix, feedback
    """
    ref_phonemes = extract_phonemes(ref_text, language)
    hyp_phonemes = extract_phonemes(hyp_text, language)
    
    per = compute_per(ref_phonemes, hyp_phonemes)
    matrix, aligned_ref, aligned_hyp = phoneme_confusion(ref_phonemes, hyp_phonemes)
    feedback = generate_feedback(ref_phonemes, hyp_phonemes, aligned_ref, aligned_hyp, language)
    
    return {
        "per": per,
        "matrix": matrix,
        "feedback": feedback,
        "ref_phonemes": ref_phonemes,
        "hyp_phonemes": hyp_phonemes,
    }