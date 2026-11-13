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
