"""
NLP processing utilities for resume and job-description analysis.

This module provides:
- Text normalization
- Technical skill extraction
- Canonical skill mapping
- Synonym-aware skill detection
- Basic NLP processing using spaCy
"""

import re
from typing import List, Set

import spacy

from src.utils.keyword_mapping import (
    SKILL_GROUPS,
    expand_keywords_with_synonyms,
    map_keyword_to_group,
)


# ============================================================
# SPACY MODEL
# ============================================================

_NLP = None


def _load_spacy_model():
    """
    Load spaCy English model once and reuse it.

    Returns:
        spaCy language model.
    """

    global _NLP

    if _NLP is not None:
        return _NLP

   try:
    _NLP = spacy.load("en_core_web_sm")

except OSError as e:
    raise RuntimeError(
        "spaCy English model is missing. "
        "Please install en_core_web_sm through requirements.txt."
    ) from e

    return _NLP


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize input text for NLP processing.

    Operations:
    - Convert Windows line endings
    - Remove excessive whitespace
    - Preserve meaningful punctuation
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace multiple newlines with one newline.
    text = re.sub(r"\n+", "\n", text)

    # Replace repeated spaces/tabs with one space.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _normalize_for_matching(text: str) -> str:
    """
    Create a normalized version of text for skill matching.

    This keeps characters such as:
    +, #, /, -, .
    
    because technical skills can contain them.

    Examples:
        C++ -> c++
        C# -> c#
        scikit-learn -> scikit-learn
        CI/CD -> ci/cd
    """

    text = text.lower()

    # Normalize different dash characters.
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _skill_pattern(skill: str) -> str:
    """
    Build a safe regex pattern for a technical skill.

    Lookarounds prevent matching a skill inside a larger word.

    Example:
        python should match:
            "Python developer"

        but should not match:
            "pythonic"
    """

    escaped = re.escape(skill.lower().strip())

    return rf"(?<!\w){escaped}(?!\w)"


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text: str) -> List[str]:
    """
    Extract technical skills from text.

    The returned skills are canonical names defined in
    keyword_mapping.py.

    Example:

        Input:
            "Experience with Python, NumPy, PyTorch and
             natural language processing."

        Output:
            [
                "nlp",
                "numpy",
                "python",
                "pytorch"
            ]
    """

    if not text:
        return []

    normalized_text = _normalize_for_matching(text)

    detected_skills: Set[str] = set()

    for canonical_skill, synonyms in SKILL_GROUPS.items():

        for synonym in synonyms:

            synonym_normalized = synonym.lower().strip()

            if not synonym_normalized:
                continue

            pattern = _skill_pattern(synonym_normalized)

            if re.search(pattern, normalized_text):
                detected_skills.add(canonical_skill)
                break

    return sorted(detected_skills)


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(
    text: str,
    top_k: int = 15,
    use_synonyms: bool = True,
) -> List[str]:
    """
    Extract important technical keywords/skills.

    For the resume screener, technical skill extraction is
    preferred over generic noun-chunk extraction.

    Args:
        text:
            Resume or job-description text.

        top_k:
            Maximum number of skills to return.

        use_synonyms:
            Kept for compatibility with the original project.

    Returns:
        List of canonical technical skills.
    """

    if not text:
        return []

    # Run spaCy so the project still uses an actual NLP pipeline.
    # This also validates that the language model is available.
    nlp = _load_spacy_model()
    _ = nlp(text)

    skills = extract_skills(text)

    if use_synonyms:
        # Convert detected skills through the canonical mapping.
        canonical_skills = set()

        for skill in skills:
            canonical = map_keyword_to_group(skill)
            canonical_skills.add(canonical)

        skills = sorted(canonical_skills)

    # Limit number of returned skills.
    return skills[:top_k]


# ============================================================
# SYNONYM-AWARE SKILL EXTRACTION
# ============================================================

def extract_skills_with_synonyms(text: str) -> List[str]:
    """
    Extract skills while explicitly using the synonym expansion
    system.

    This function is useful when we want to inspect all
    recognized forms of a skill.
    """

    if not text:
        return []

    detected = extract_skills(text)

    if not detected:
        return []

    expanded = expand_keywords_with_synonyms(detected)

    canonical_skills: Set[str] = set()

    for keyword in expanded:

        canonical = map_keyword_to_group(keyword)

        if canonical in SKILL_GROUPS:
            canonical_skills.add(canonical)

    return sorted(canonical_skills)


# ============================================================
# BASIC NLP INFORMATION
# ============================================================

def extract_entities(text: str) -> List[str]:
    """
    Extract named entities using spaCy.

    This can later be used for:
    - Organizations
    - Locations
    - Dates
    - Companies
    - Other entities

    Returns:
        Unique entity strings.
    """

    if not text:
        return []

    nlp = _load_spacy_model()
    doc = nlp(text)

    entities = {
        entity.text.strip()
        for entity in doc.ents
        if entity.text.strip()
    }

    return sorted(entities)


# ============================================================
# TEXT STATISTICS
# ============================================================

def get_text_statistics(text: str) -> dict:
    """
    Return basic NLP/text statistics.

    Useful for debugging and future dashboard features.
    """

    if not text:
        return {
            "characters": 0,
            "words": 0,
            "sentences": 0,
            "skills": 0,
        }

    nlp = _load_spacy_model()
    doc = nlp(text)

    words = [
        token.text
        for token in doc
        if not token.is_space
    ]

    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(list(doc.sents)),
        "skills": len(extract_skills(text)),
    }
