"""
Scoring engine for the AI Resume Screener.

This module calculates:
- Semantic similarity
- Technical skill coverage
- Job-relevant experience match
- Education match
- Certification match
- Overall candidate score
- Matched skills
- Missing skills
- Additional skills

Important:
Experience matching is job-specific.

A candidate having 5 years of total experience does NOT
automatically mean that they have 5 years of relevant
experience for every job.

N/A handling:
If a job description does not require a particular criterion,
that criterion is marked as N/A and does not affect the
candidate's overall score.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.keyword_mapping import (
    EDUCATION_KEYWORDS,
    map_keyword_to_group,
)


# ============================================================
# SECTION DEFINITIONS
# ============================================================

EXPERIENCE_HEADERS = {
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "employment experience",
    "work history",
    "career history",
    "professional background",
    "internship",
    "internships",
    "internship experience",
    "internship experience and training",
    "work experience and internships",
    "experience and internships",
    "work experience & internships",
    "experience & internships",
}

NON_EXPERIENCE_HEADERS = {
    "education",
    "academic background",
    "academic qualifications",
    "qualifications",
    "projects",
    "project",
    "personal projects",
    "academic projects",
    "certifications",
    "certification",
    "professional certifications",
    "professional certification",
    "licenses",
    "licenses and certifications",
    "licenses & certifications",
    "certificates",
    "certificates and certifications",
    "training",
    "trainings",
    "courses",
    "coursework",
    "skills",
    "technical skills",
    "technical expertise",
    "achievements",
    "awards",
    "languages",
    "interests",
    "hobbies",
    "summary",
    "professional summary",
    "objective",
    "career objective",
    "profile",
    "contact",
    "references",
    "publications",
    "activities",
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def _normalize_skill(skill: str) -> str:
    """
    Normalize a skill name for comparison.
    """

    if not skill:
        return ""

    return " ".join(
        skill.lower().strip().split()
    )


# Skills that should count as the same requirement when comparing a
# JD with a resume. A JD asking for "Git and GitHub" is satisfied by a
# resume that lists GitHub, and vice-versa.
SKILL_ALIASES = {
    "github": "git",
}


def _canonicalize_skills(
    skills: List[str],
) -> Set[str]:
    """
    Convert detected skills into canonical skill names.

    Examples:

        sklearn -> scikit_learn
        py torch -> pytorch
        natural language processing -> nlp
    """

    canonical_skills: Set[str] = set()

    for skill in skills:

        normalized = _normalize_skill(skill)

        if not normalized:
            continue

        canonical = map_keyword_to_group(
            normalized
        )

        canonical = SKILL_ALIASES.get(
            canonical,
            canonical,
        )

        canonical_skills.add(canonical)

    return canonical_skills


# ============================================================
# TEXT / SECTION HELPERS
# ============================================================

def _normalize_header(line: str) -> str:
    """
    Normalize a possible resume section heading.
    """

    line = line.lower().strip()

    line = re.sub(
        r"^[\s•\-–—*#:]+",
        "",
        line,
    )

    line = re.sub(
        r"[\s:]+$",
        "",
        line,
    )

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line


def _is_experience_header(
    line: str,
) -> bool:
    """
    Determine whether a line is an experience heading.
    """

    return (
        _normalize_header(line)
        in EXPERIENCE_HEADERS
    )


def _is_non_experience_header(
    line: str,
) -> bool:
    """
    Determine whether a line begins a non-experience section.
    """

    return (
        _normalize_header(line)
        in NON_EXPERIENCE_HEADERS
    )


def _extract_experience_sections(
    text: str,
) -> List[str]:
    """
    Extract only employment/experience sections from a resume.

    Education, projects, certifications, training, etc.
    are excluded.
    """

    if not text:
        return []

    lines = text.splitlines()

    sections = []
    current_section = []

    inside_experience = False

    for line in lines:

        if _is_experience_header(line):

            if (
                inside_experience
                and current_section
            ):
                sections.append(
                    "\n".join(
                        current_section
                    )
                )

            current_section = []
            inside_experience = True

            continue

        if _is_non_experience_header(line):

            if (
                inside_experience
                and current_section
            ):
                sections.append(
                    "\n".join(
                        current_section
                    )
                )

            current_section = []
            inside_experience = False

            continue

        if inside_experience:

            current_section.append(line)

    if (
        inside_experience
        and current_section
    ):
        sections.append(
            "\n".join(
                current_section
            )
        )

    return [
        section
        for section in sections
        if section.strip()
    ]


# ============================================================
# JOB DESCRIPTION EXPERIENCE EXTRACTION
# ============================================================

def _extract_required_experience(
    jd_text: str,
) -> Optional[float]:
    """
    Extract required years of experience from a JD.

    Examples:

        2 years experience
        3+ years experience
        minimum 5 years
        at least 4 years of experience
        2 yrs of experience
    """

    if not jd_text:
        return None

    jd_lower = jd_text.lower()

    # --------------------------------------------------------
    # Experience ranges
    # --------------------------------------------------------
    # Check ranges BEFORE single-number patterns.
    # Otherwise "0-2 years" may be incorrectly read as "2 years".
    # For a range, the lower bound is the minimum required experience.
    range_patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",

        r"(?:experience|exp)\s*[:=-]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
    ]

    range_values = []

    for pattern in range_patterns:
        for match in re.finditer(pattern, jd_lower):
            try:
                lower = float(match.group(1))
                upper = float(match.group(2))
                if 0 <= lower <= 50 and 0 <= upper <= 50 and lower <= upper:
                    range_values.append(lower)
            except (ValueError, TypeError):
                continue

    if range_values:
        return min(range_values)

    # --------------------------------------------------------
    # Single-value experience requirements
    # --------------------------------------------------------
    patterns = [
        # Experience: 2 years / Experience - 2 years / Exp: 2 yrs
        # This is the common JD format used by the Streamlit app.
        r"(?:experience|exp)\s*[:=-]\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*(?:years?|yrs?)",

        # 2 years experience / 2+ years professional experience
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?"
        r"\s*(?:of\s*)?"
        r"(?:professional\s*)?"
        r"(?:experience|exp)",

        # minimum 3 years
        r"(?:minimum|min)"
        r"\s+(?:of\s+)?"
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?",

        # at least 3 years
        r"at\s+least\s+"
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?",

        # 3 yrs experience
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*yrs?"
        r"\s*(?:of\s*)?"
        r"(?:experience|exp)",
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            jd_lower,
        )

        for match in matches:

            try:

                value = float(match)

                if 0 <= value <= 50:
                    values.append(value)

            except (
                ValueError,
                TypeError,
            ):
                continue

    if not values:
        return None

    return max(values)


# ============================================================
# JD RELEVANCE TERMS
# ============================================================

def _extract_jd_relevance_terms(
    jd_text: str,
    jd_keywords: List[str],
) -> List[str]:
    """
    Extract terms that can be used to determine whether an
    employment position is relevant to the current JD.

    This combines:
    - canonical JD skills
    - important role words
    - important technology terms
    """

    terms: Set[str] = set()

    for skill in jd_keywords:

        normalized = _normalize_skill(skill)

        if normalized:
            terms.add(normalized)

        canonical = map_keyword_to_group(
            normalized
        )

        if canonical:
            terms.add(canonical)

    role_terms = [
        "engineer",
        "developer",
        "scientist",
        "analyst",
        "architect",
        "administrator",
        "consultant",
        "manager",
        "designer",
        "machine learning",
        "data science",
        "data scientist",
        "software development",
        "backend",
        "frontend",
        "full stack",
        "devops",
        "cloud",
        "cybersecurity",
        "artificial intelligence",
        "nlp",
        "natural language processing",
    ]

    jd_lower = jd_text.lower()

    for term in role_terms:

        if _contains_keyword(
            jd_lower,
            term,
        ):
            terms.add(term)

    return sorted(
        terms,
        key=len,
        reverse=True,
    )


# ============================================================
# SKILL / KEYWORD COVERAGE
# ============================================================

def _keyword_coverage(
    jd_keywords: List[str],
    resume_keywords: List[str],
) -> float:
    """
    Calculate technical skill coverage.

    Formula:

        matched JD skills / total JD skills
    """

    jd_skills = _canonicalize_skills(
        jd_keywords
    )

    resume_skills = _canonicalize_skills(
        resume_keywords
    )

    if not jd_skills:
        return 0.0

    matched = jd_skills.intersection(
        resume_skills
    )

    return (
        len(matched)
        / float(len(jd_skills))
    )


def _get_matched_skills(
    jd_keywords: List[str],
    resume_keywords: List[str],
) -> List[str]:
    """
    Return JD skills found in the resume.
    """

    jd_skills = _canonicalize_skills(
        jd_keywords
    )

    resume_skills = _canonicalize_skills(
        resume_keywords
    )

    return sorted(
        jd_skills.intersection(
            resume_skills
        )
    )


def _get_missing_skills(
    jd_keywords: List[str],
    resume_keywords: List[str],
) -> List[str]:
    """
    Return JD skills missing from the resume.
    """

    jd_skills = _canonicalize_skills(
        jd_keywords
    )

    resume_skills = _canonicalize_skills(
        resume_keywords
    )

    return sorted(
        jd_skills - resume_skills
    )


def _get_additional_skills(
    jd_keywords: List[str],
    resume_keywords: List[str],
) -> List[str]:
    """
    Return resume skills that are not required by the JD.
    """

    jd_skills = _canonicalize_skills(
        jd_keywords
    )

    resume_skills = _canonicalize_skills(
        resume_keywords
    )

    return sorted(
        resume_skills - jd_skills
    )


# ============================================================
# EXPERIENCE DATE EXTRACTION
# ============================================================

def _extract_experience_intervals(
    text: str,
) -> List[Tuple[int, int]]:
    """
    Extract employment date ranges from an experience section.

    Supports:

        Jan 2022 - Dec 2024
        January 2022 - December 2024
        Jan 2022 - Present
        2020 - 2024
        2022 - Present

    Internally dates are represented as:
        year * 12 + month
    """

    if not text:
        return []

    now = datetime.now()

    current_year = now.year
    current_month = now.month

    intervals = []

    months = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    month_pattern = (
        r"(?:"
        r"jan(?:uary)?|"
        r"feb(?:ruary)?|"
        r"mar(?:ch)?|"
        r"apr(?:il)?|"
        r"may|"
        r"jun(?:e)?|"
        r"jul(?:y)?|"
        r"aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|"
        r"nov(?:ember)?|"
        r"dec(?:ember)?"
        r")"
    )

    # --------------------------------------------------------
    # Month + year ranges
    # --------------------------------------------------------

    month_range_pattern = re.compile(
        rf"\b"
        rf"({month_pattern})"
        rf"\s+(\d{{4}})"
        rf"\s*(?:-|–|—|to)\s*"
        rf"(?:"
        rf"({month_pattern})"
        rf"\s+(\d{{4}})"
        rf"|"
        rf"(present|current|now)"
        rf")"
        rf"\b",
        re.IGNORECASE,
    )

    consumed_spans = []

    for match in month_range_pattern.finditer(
        text.lower()
    ):

        try:

            start_month = months[
                match.group(1).lower()
            ]

            start_year = int(
                match.group(2)
            )

            start_index = (
                start_year * 12
                + start_month
            )

            if match.group(5):

                end_index = (
                    current_year * 12
                    + current_month
                )

            else:

                end_month = months[
                    match.group(3).lower()
                ]

                end_year = int(
                    match.group(4)
                )

                end_index = (
                    end_year * 12
                    + end_month
                )

            if (
                1950 <= start_year <= current_year
                and start_index <= end_index
            ):

                intervals.append(
                    (
                        start_index,
                        end_index,
                    )
                )

                consumed_spans.append(
                    match.span()
                )

        except (
            ValueError,
            KeyError,
            IndexError,
        ):
            continue

    # --------------------------------------------------------
    # Year-only ranges
    # --------------------------------------------------------

    year_range_pattern = re.compile(
        r"\b"
        r"(19\d{2}|20\d{2})"
        r"\s*(?:-|–|—|to)\s*"
        r"(present|current|now|19\d{2}|20\d{2})"
        r"\b",
        re.IGNORECASE,
    )

    for match in year_range_pattern.finditer(
        text.lower()
    ):

        overlapping = False

        for span_start, span_end in consumed_spans:

            if (
                match.start() < span_end
                and match.end() > span_start
            ):

                overlapping = True
                break

        if overlapping:
            continue

        try:

            start_year = int(
                match.group(1)
            )

            end_value = (
                match.group(2).lower()
            )

            if end_value in {
                "present",
                "current",
                "now",
            }:

                end_year = current_year

            else:

                end_year = int(
                    end_value
                )

            if (
                1950 <= start_year <= current_year
                and start_year <= end_year
            ):

                start_index = (
                    start_year * 12
                    + 1
                )

                end_index = (
                    end_year * 12
                    + 12
                )

                intervals.append(
                    (
                        start_index,
                        end_index,
                    )
                )

        except (
            ValueError,
            IndexError,
        ):
            continue

    return intervals


# ============================================================
# INTERVAL MERGING
# ============================================================

def _merge_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Merge overlapping employment intervals.
    """

    if not intervals:
        return []

    sorted_intervals = sorted(
        intervals
    )

    merged = [
        list(sorted_intervals[0])
    ]

    for start, end in sorted_intervals[1:]:

        previous_end = merged[-1][1]

        if start <= previous_end + 1:

            merged[-1][1] = max(
                previous_end,
                end,
            )

        else:

            merged.append(
                [
                    start,
                    end,
                ]
            )

    return [
        (
            start,
            end,
        )
        for start, end in merged
    ]


def _interval_months(
    intervals: List[Tuple[int, int]],
) -> int:
    """
    Return total number of months in intervals.
    """

    if not intervals:
        return 0

    merged = _merge_intervals(
        intervals
    )

    total_months = 0

    for start, end in merged:

        duration = end - start + 1

        if duration > 0:
            total_months += duration

    return total_months


# ============================================================
# KEYWORD MATCHING
# ============================================================

def _contains_keyword(
    text: str,
    keyword: str,
) -> bool:
    """
    Check whether a word or phrase occurs meaningfully.
    """

    if not text or not keyword:
        return False

    text = text.lower()
    keyword = keyword.lower().strip()

    keyword_pattern = keyword.replace(
        "_",
        r"[\s_-]+",
    )

    pattern = (
        rf"(?<!\w)"
        rf"{keyword_pattern}"
        rf"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text,
        )
    )


def _calculate_line_relevance(
    line: str,
    relevance_terms: List[str],
) -> float:
    """
    Estimate how strongly an employment line relates to
    the current job description.

    Returns a value between 0 and 1.
    """

    if not line:
        return 0.0

    line_lower = line.lower()

    if not relevance_terms:
        return 0.0

    matches = 0

    for term in relevance_terms:

        if _contains_keyword(
            line_lower,
            term,
        ):
            matches += 1

    if matches == 0:
        return 0.0

    if matches == 1:
        return 0.6

    if matches == 2:
        return 0.8

    return 1.0


# ============================================================
# JOB-SPECIFIC EXPERIENCE
# ============================================================

def _estimate_relevant_experience(
    jd_text: str,
    jd_keywords: List[str],
    resume_text: str,
    resume_exp_years: Optional[float],
    resume_roles: Optional[List[str]] = None,
) -> Tuple[Optional[float], str]:
    """
    Estimate experience relevant to the current JD.

    Strategy:

    1. Find employment/experience sections.
    2. Extract employment date ranges.
    3. Determine whether the experience contains
       JD-relevant role/skill evidence.
    4. Count relevant employment intervals.
    5. Merge overlapping relevant intervals.

    If resume text is unavailable, use a conservative fallback
    based on extracted job roles.
    """

    if resume_exp_years is None:

        return (
            None,
            "No reliable experience information found.",
        )

    required_years = _extract_required_experience(
        jd_text
    )

    if not resume_text:

        roles = resume_roles or []

        relevance_terms = (
            _extract_jd_relevance_terms(
                jd_text,
                jd_keywords,
            )
        )

        role_relevance = 0.0

        for role in roles:

            score = _calculate_line_relevance(
                role,
                relevance_terms,
            )

            role_relevance = max(
                role_relevance,
                score,
            )

        if role_relevance == 0.0:

            return (
                0.0,
                "No job-relevant role evidence found.",
            )

        estimated = (
            resume_exp_years
            * role_relevance
        )

        return (
            round(
                estimated,
                1,
            ),
            "Relevant experience estimated from extracted job roles.",
        )

    experience_sections = (
        _extract_experience_sections(
            resume_text
        )
    )

    if not experience_sections:

        return (
            0.0,
            "No identifiable employment/experience section found.",
        )

    relevance_terms = (
        _extract_jd_relevance_terms(
            jd_text,
            jd_keywords,
        )
    )

    relevant_intervals = []

    for section in experience_sections:

        lines = section.splitlines()

        intervals = (
            _extract_experience_intervals(
                section
            )
        )

        if not intervals:
            continue

        section_text = section.lower()

        relevance_score = (
            _calculate_line_relevance(
                section_text,
                relevance_terms,
            )
        )

        line_scores = []

        for line in lines:

            score = _calculate_line_relevance(
                line,
                relevance_terms,
            )

            if score > 0:

                line_scores.append(
                    score
                )

        if line_scores:

            relevance_score = max(
                relevance_score,
                max(line_scores),
            )

        if relevance_score >= 0.6:

            relevant_intervals.extend(
                intervals
            )

    relevant_months = _interval_months(
        relevant_intervals
    )

    if relevant_months <= 0:

        return (
            0.0,
            "No clearly job-relevant employment period detected.",
        )

    relevant_years = (
        relevant_months / 12.0
    )

    # Never report more relevant experience than total
    # extracted candidate experience.
    relevant_years = min(
        relevant_years,
        resume_exp_years,
    )

    relevant_years = round(
        relevant_years,
        1,
    )

    if required_years is None:

        explanation = (
            f"Estimated {relevant_years:.1f} "
            "years of experience relevant to the job."
        )

    else:

        explanation = (
            f"Estimated {relevant_years:.1f} "
            f"years relevant to a requirement of "
            f"{required_years:.1f} years."
        )

    return (
        relevant_years,
        explanation,
    )


# ============================================================
# EXPERIENCE MATCH
# ============================================================

def _experience_match(
    jd_text: str,
    resume_exp_years: Optional[float],
    resume_text: str = "",
    jd_keywords: Optional[List[str]] = None,
    resume_roles: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Calculate job-specific experience score.

    Returns:

        (
            score,
            relevant_experience_years,
            explanation
        )

    Score:

        1.0 -> requirement satisfied
        0.8 -> candidate has >= 80% of requirement
        0.6 -> candidate has >= 60% of requirement
        proportional -> candidate has less
        None -> no experience requirement
    """

    required_years = _extract_required_experience(
        jd_text
    )

    # --------------------------------------------------------
    # N/A
    # --------------------------------------------------------

    if required_years is None:

        return (
            None,
            None,
            "No explicit experience requirement in the job description.",
        )

    # --------------------------------------------------------
    # No minimum experience (e.g. "0-1 year", "freshers welcome")
    # --------------------------------------------------------
    # Every candidate satisfies a minimum of 0 years, so this
    # criterion cannot separate candidates. Treat it as N/A so it is
    # excluded from the weighted score instead of penalising freshers.

    if required_years <= 0:

        return (
            None,
            None,
            "Job description has no minimum experience requirement "
            "(freshers welcome); experience is not scored.",
        )

    # --------------------------------------------------------
    # Candidate experience unavailable
    # --------------------------------------------------------

    if resume_exp_years is None:

        return (
            0.0,
            None,
            "Candidate experience could not be reliably determined.",
        )

    # --------------------------------------------------------
    # Estimate relevant experience
    # --------------------------------------------------------

    (
        relevant_years,
        explanation,
    ) = _estimate_relevant_experience(
        jd_text=jd_text,
        jd_keywords=jd_keywords or [],
        resume_text=resume_text,
        resume_exp_years=resume_exp_years,
        resume_roles=resume_roles,
    )

    if relevant_years is None:

        return (
            0.0,
            None,
            explanation,
        )

    # --------------------------------------------------------
    # Requirement satisfied
    # --------------------------------------------------------

    if relevant_years >= required_years:

        return (
            1.0,
            relevant_years,
            explanation
            + " Requirement satisfied.",
        )

    # --------------------------------------------------------
    # No relevant experience
    # --------------------------------------------------------

    if relevant_years <= 0:

        return (
            0.0,
            0.0,
            explanation
            + " Requirement not satisfied.",
        )

    ratio = (
        relevant_years
        / required_years
    )

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    if ratio >= 0.8:

        score = 0.8

    elif ratio >= 0.6:

        score = 0.6

    else:

        score = max(
            0.0,
            min(
                1.0,
                ratio,
            ),
        )

    return (
        score,
        relevant_years,
        explanation
        + " Requirement partially satisfied.",
    )


# ============================================================
# EDUCATION MATCHING
# ============================================================

def _required_education(
    jd_text: str,
) -> Dict[str, bool]:
    """
    Detect education requirements from the JD.
    """

    requirements = {
        "phd": False,
        "masters": False,
        "bachelors": False,
        "diploma": False,
    }

    if not jd_text:
        return requirements

    jd_lower = jd_text.lower()

    for degree_type, keywords in (
        EDUCATION_KEYWORDS.items()
    ):

        for keyword in keywords:

            if _contains_keyword(
                jd_lower,
                keyword,
            ):

                requirements[
                    degree_type
                ] = True

                break

    return requirements


def _education_match(
    jd_text: str,
    resume_education: Dict[str, bool],
) -> Tuple[Optional[float], bool]:
    """
    Compare candidate education with JD requirements.

    Returns:

        (score, is_applicable)

    Score:

        1.0 -> requirement satisfied
        0.9 -> higher qualification satisfies diploma
        0.0 -> requirement not satisfied
        None -> no education requirement
    """

    requirements = _required_education(
        jd_text
    )

    # --------------------------------------------------------
    # N/A
    # --------------------------------------------------------

    if not any(
        requirements.values()
    ):

        return (
            None,
            False,
        )

    resume_education = (
        resume_education or {}
    )

    # --------------------------------------------------------
    # PhD
    # --------------------------------------------------------

    if requirements["phd"]:

        if resume_education.get(
            "phd",
            False,
        ):

            return (
                1.0,
                True,
            )

        return (
            0.0,
            True,
        )

    # --------------------------------------------------------
    # Master's
    # --------------------------------------------------------

    if requirements["masters"]:

        if resume_education.get(
            "masters",
            False,
        ):

            return (
                1.0,
                True,
            )

        if resume_education.get(
            "phd",
            False,
        ):

            return (
                1.0,
                True,
            )

        return (
            0.0,
            True,
        )

    # --------------------------------------------------------
    # Bachelor's
    # --------------------------------------------------------

    if requirements["bachelors"]:

        if resume_education.get(
            "bachelors",
            False,
        ):

            return (
                1.0,
                True,
            )

        if resume_education.get(
            "masters",
            False,
        ):

            return (
                1.0,
                True,
            )

        if resume_education.get(
            "phd",
            False,
        ):

            return (
                1.0,
                True,
            )

        return (
            0.0,
            True,
        )

    # --------------------------------------------------------
    # Diploma
    # --------------------------------------------------------

    if requirements["diploma"]:

        if resume_education.get(
            "diploma",
            False,
        ):

            return (
                1.0,
                True,
            )

        if resume_education.get(
            "bachelors",
            False,
        ):

            return (
                0.9,
                True,
            )

        if resume_education.get(
            "masters",
            False,
        ):

            return (
                0.9,
                True,
            )

        if resume_education.get(
            "phd",
            False,
        ):

            return (
                0.9,
                True,
            )

        return (
            0.0,
            True,
        )

    return (
        None,
        False,
    )


# ============================================================
# CERTIFICATION MATCHING
# ============================================================

def _jd_requires_certification(
    jd_text: str,
) -> bool:
    """
    Determine whether the JD explicitly requires or prefers
    a certification.

    This intentionally looks for certification-related
    language rather than assuming every JD requires one.
    """

    if not jd_text:
        return False

    jd_lower = jd_text.lower()

    certification_patterns = [
        r"\bcertified\b",
        r"\bcertification\b",
        r"\bcertificate\b",
        r"\bcertifications\b",
        r"\blicense\b",
        r"\blicensed\b",
    ]

    return any(
        re.search(
            pattern,
            jd_lower,
        )
        for pattern in certification_patterns
    )


def _certification_match(
    jd_text: str,
    resume_certs: List[str],
) -> Tuple[Optional[float], bool]:
    """
    Compare certifications mentioned in the JD and resume.

    Returns:

        (score, is_applicable)

    Score:

        1.0 -> matching certification evidence
        0.3 -> certifications exist but no specific match
        0.0 -> requirement exists but candidate has none
        None -> certification not required
    """

    # --------------------------------------------------------
    # N/A
    # --------------------------------------------------------

    if not _jd_requires_certification(
        jd_text
    ):

        return (
            None,
            False,
        )

    # --------------------------------------------------------
    # Certification required but candidate has none
    # --------------------------------------------------------

    if not resume_certs:

        return (
            0.0,
            True,
        )

    jd_lower = jd_text.lower()

    jd_words = set(
        re.findall(
            r"[a-zA-Z0-9+#.-]+",
            jd_lower,
        )
    )

    generic_words = {
        "certified",
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "cert",
        "required",
        "preferred",
        "experience",
        "years",
        "year",
        "minimum",
    }

    jd_words -= generic_words

    for cert in resume_certs:

        cert_words = set(
            re.findall(
                r"[a-zA-Z0-9+#.-]+",
                str(cert).lower(),
            )
        )

        cert_words -= generic_words

        if jd_words.intersection(
            cert_words
        ):

            return (
                1.0,
                True,
            )

    return (
        0.3,
        True,
    )


# ============================================================
# SCORE EXPLANATION
# ============================================================

def _generate_score_explanation(
    similarity: float,
    keyword_coverage: float,
    experience_score: Optional[float],
    education_score: Optional[float],
    certification_score: Optional[float],
) -> str:
    """
    Generate a short human-readable score explanation.
    """

    reasons = []

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    if similarity >= 0.75:

        reasons.append(
            "strong semantic similarity"
        )

    elif similarity >= 0.50:

        reasons.append(
            "moderate semantic similarity"
        )

    else:

        reasons.append(
            "low semantic similarity"
        )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    if keyword_coverage >= 0.75:

        reasons.append(
            "strong skill coverage"
        )

    elif keyword_coverage >= 0.50:

        reasons.append(
            "moderate skill coverage"
        )

    else:

        reasons.append(
            "low skill coverage"
        )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    if experience_score is None:

        reasons.append(
            "experience requirement not specified"
        )

    elif experience_score >= 1.0:

        reasons.append(
            "relevant experience requirement satisfied"
        )

    elif experience_score == 0.0:

        reasons.append(
            "relevant experience requirement not satisfied"
        )

    else:

        reasons.append(
            "partial relevant experience match"
        )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    if education_score is None:

        reasons.append(
            "education requirement not specified"
        )

    elif education_score >= 1.0:

        reasons.append(
            "education requirement satisfied"
        )

    elif education_score == 0.0:

        reasons.append(
            "education requirement not satisfied"
        )

    # --------------------------------------------------------
    # Certifications
    # --------------------------------------------------------

    if certification_score is None:

        reasons.append(
            "certification requirement not specified"
        )

    elif certification_score >= 1.0:

        reasons.append(
            "certification requirement satisfied"
        )

    elif certification_score == 0.0:

        reasons.append(
            "certification requirement not satisfied"
        )

    # --------------------------------------------------------
    # Final explanation
    # --------------------------------------------------------

    if not reasons:

        return (
            "No scoring explanation available."
        )

    return (
        ", ".join(reasons).capitalize()
        + "."
    )


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def compute_scores(
    jd_text: str,
    jd_keywords: List[str],
    resumes: List[Dict],
    sim_matrix: np.ndarray,
    weight_similarity: float,
    weight_keywords: float,
    weight_experience: float = 0.0,
    weight_education: float = 0.0,
    weight_certifications: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate scores for all resumes.

    Final score is a weighted average of applicable criteria:

        Semantic Similarity
        Technical Skill Coverage
        Job-Relevant Experience
        Education
        Certification

    If a criterion is not required by the JD, it is excluded
    from the final weighted score.

    Returns:
        DataFrame sorted by overall score.
    """

    # --------------------------------------------------------
    # Safety check for similarity matrix
    # --------------------------------------------------------

    if sim_matrix is None:

        sim_matrix = np.zeros(
            (
                1,
                len(resumes) + 1,
            )
        )

    try:

        jd_to_res_sims = (
            sim_matrix[0, 1:].tolist()
        )

    except (
        IndexError,
        AttributeError,
    ):

        jd_to_res_sims = [
            0.0
            for _ in resumes
        ]

    rows = []

    # ========================================================
    # PROCESS EACH RESUME
    # ========================================================

    for idx, resume in enumerate(
        resumes
    ):

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        if idx < len(
            jd_to_res_sims
        ):

            similarity = float(
                jd_to_res_sims[idx]
            )

        else:

            similarity = 0.0

        similarity = max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

        # ----------------------------------------------------
        # Skills
        # ----------------------------------------------------

        resume_keywords = resume.get(
            "keywords",
            [],
        )

        keyword_coverage = (
            _keyword_coverage(
                jd_keywords,
                resume_keywords,
            )
        )

        matched_skills = (
            _get_matched_skills(
                jd_keywords,
                resume_keywords,
            )
        )

        missing_skills = (
            _get_missing_skills(
                jd_keywords,
                resume_keywords,
            )
        )

        additional_skills = (
            _get_additional_skills(
                jd_keywords,
                resume_keywords,
            )
        )

        # ----------------------------------------------------
        # Experience
        # ----------------------------------------------------

        experience_years = resume.get(
            "experience_years"
        )

        resume_text = resume.get(
            "text",
            "",
        )

        resume_roles = resume.get(
            "recent_roles",
            [],
        )

        (
            experience_score,
            relevant_experience_years,
            experience_explanation,
        ) = _experience_match(
            jd_text=jd_text,
            resume_exp_years=experience_years,
            resume_text=resume_text,
            jd_keywords=jd_keywords,
            resume_roles=resume_roles,
        )

        # ----------------------------------------------------
        # Education
        # ----------------------------------------------------

        education = resume.get(
            "education",
            {},
        )

        (
            education_score,
            education_applicable,
        ) = _education_match(
            jd_text,
            education,
        )

        # ----------------------------------------------------
        # Certifications
        # ----------------------------------------------------

        certifications = resume.get(
            "certifications",
            [],
        )

        (
            certification_score,
            certification_applicable,
        ) = _certification_match(
            jd_text,
            certifications,
        )

        # ====================================================
        # APPLICABLE WEIGHT CALCULATION
        # ====================================================

        applicable_weight = 0.0
        weighted_score = 0.0

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        if weight_similarity > 0:

            applicable_weight += (
                weight_similarity
            )

            weighted_score += (
                weight_similarity
                * similarity
            )

        # ----------------------------------------------------
        # Skill coverage
        # ----------------------------------------------------

        if weight_keywords > 0:

            applicable_weight += (
                weight_keywords
            )

            weighted_score += (
                weight_keywords
                * keyword_coverage
            )

        # ----------------------------------------------------
        # Experience
        # ----------------------------------------------------

        if (
            weight_experience > 0
            and experience_score is not None
        ):

            applicable_weight += (
                weight_experience
            )

            weighted_score += (
                weight_experience
                * experience_score
            )

        # ----------------------------------------------------
        # Education
        # ----------------------------------------------------

        if (
            weight_education > 0
            and education_applicable
            and education_score is not None
        ):

            applicable_weight += (
                weight_education
            )

            weighted_score += (
                weight_education
                * education_score
            )

        # ----------------------------------------------------
        # Certifications
        # ----------------------------------------------------

        if (
            weight_certifications > 0
            and certification_applicable
            and certification_score is not None
        ):

            applicable_weight += (
                weight_certifications
            )

            weighted_score += (
                weight_certifications
                * certification_score
            )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        if applicable_weight > 0:

            total_score = (
                weighted_score
                / applicable_weight
            )

        else:

            total_score = 0.0

        total_score = float(
            max(
                0.0,
                min(
                    1.0,
                    total_score,
                ),
            )
        )

        # ====================================================
        # DISPLAY VALUES
        # ====================================================

        if experience_years is not None:

            experience_years_str = (
                f"{experience_years:.1f}"
            )

        else:

            experience_years_str = "N/A"

        if relevant_experience_years is not None:

            relevant_experience_str = (
                f"{relevant_experience_years:.1f}"
            )

        else:

            relevant_experience_str = "N/A"

        # ----------------------------------------------------
        # Education display
        # ----------------------------------------------------

        education_list = []

        if education.get("phd"):

            education_list.append(
                "PhD"
            )

        if education.get("masters"):

            education_list.append(
                "Masters"
            )

        if education.get("bachelors"):

            education_list.append(
                "Bachelors"
            )

        if education.get("diploma"):

            education_list.append(
                "Diploma"
            )

        education_str = (
            ", ".join(
                education_list
            )
            if education_list
            else "Not specified"
        )

        # ----------------------------------------------------
        # Certification display
        # ----------------------------------------------------

        certification_count = len(
            certifications
        )

        # ----------------------------------------------------
        # Human-readable score explanation
        # ----------------------------------------------------

        score_explanation = (
            _generate_score_explanation(
                similarity=similarity,
                keyword_coverage=keyword_coverage,
                experience_score=experience_score,
                education_score=education_score,
                certification_score=certification_score,
            )
        )

        # ====================================================
        # STORE RESULT
        # ====================================================

        rows.append({

            "filename":
                resume.get(
                    "filename",
                    "Unknown",
                ),

            "similarity":
                similarity,

            "keyword_coverage":
                keyword_coverage,

            "experience_score":
                experience_score,

            "education_score":
                education_score,

            "certification_score":
                certification_score,

            "total_score":
                total_score,

            "experience_years":
                experience_years_str,

            "relevant_experience_years":
                relevant_experience_str,

            "education":
                education_str,

            "certifications":
                certification_count,

            "matched_keywords":
                ", ".join(
                    matched_skills[:20]
                ),

            "matched_skills":
                ", ".join(
                    matched_skills
                ),

            "missing_skills":
                ", ".join(
                    missing_skills
                ),

            "additional_skills":
                ", ".join(
                    additional_skills
                ),

            "experience_explanation":
                experience_explanation,

            "score_explanation":
                score_explanation,

            # Useful for UI logic later
            "education_applicable":
                education_applicable,

            "certification_applicable":
                certification_applicable,
        })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        return df

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        [
            "total_score",
            "similarity",
            "keyword_coverage",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # RANK
    # ========================================================

    df.insert(
        0,
        "rank",
        df.index + 1,
    )

    return df
