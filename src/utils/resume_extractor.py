"""
Extract structured information from resume text.

Features extracted:
- Total professional experience in years
- Education level
- Certifications
- Recent job roles

Important:
- Experience dates are extracted only from employment/experience sections.
- Education, project, training and certification dates are not counted
  as professional experience.
- Certification extraction is section-aware and avoids capturing
  unrelated resume sections.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.utils.keyword_mapping import EDUCATION_KEYWORDS


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
}

CERTIFICATION_HEADERS = {
    "certifications",
    "certification",
    "professional certifications",
    "professional certification",
    "licenses and certifications",
    "licenses & certifications",
    "certificates",
    "certificates and certifications",
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
# SECTION HELPERS
# ============================================================

def _normalize_header(line: str) -> str:
    """
    Normalize a possible resume section heading.
    """

    line = line.lower().strip()

    # Remove bullets/decorative characters from beginning.
    line = re.sub(
        r"^[\s•\-–—*#:]+",
        "",
        line,
    )

    # Remove trailing colon.
    line = re.sub(
        r"[\s:]+$",
        "",
        line,
    )

    # Normalize whitespace.
    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line


def _is_experience_header(line: str) -> bool:
    """
    Return True if the line is an employment/experience heading.
    """

    return _normalize_header(line) in EXPERIENCE_HEADERS


def _is_certification_header(line: str) -> bool:
    """
    Return True if the line is a certification heading.
    """

    return _normalize_header(line) in CERTIFICATION_HEADERS


def _is_non_experience_header(line: str) -> bool:
    """
    Return True if the line begins a section that should not
    contribute to professional experience.
    """

    header = _normalize_header(line)

    return header in NON_EXPERIENCE_HEADERS


# ============================================================
# EXPERIENCE SECTION EXTRACTION
# ============================================================

def _extract_experience_sections(text: str) -> List[str]:
    """
    Extract only employment/experience sections.

    Example:

        Experience
        Software Engineer
        Jan 2022 - Dec 2024

        Education
        B.Tech
        2020 - 2024

    Only the Experience section is returned.
    """

    if not text:
        return []

    lines = text.splitlines()

    sections = []
    current_section = []

    inside_experience = False

    for line in lines:

        # ----------------------------------------------------
        # Experience section starts
        # ----------------------------------------------------

        if _is_experience_header(line):

            if inside_experience and current_section:
                sections.append(
                    "\n".join(current_section)
                )

            current_section = []
            inside_experience = True
            continue

        # ----------------------------------------------------
        # Any other known section starts
        # ----------------------------------------------------

        if _is_non_experience_header(line):

            if inside_experience and current_section:
                sections.append(
                    "\n".join(current_section)
                )

            current_section = []
            inside_experience = False
            continue

        # ----------------------------------------------------
        # Store lines belonging to experience
        # ----------------------------------------------------

        if inside_experience:
            current_section.append(line)

    # --------------------------------------------------------
    # Save final section
    # --------------------------------------------------------

    if inside_experience and current_section:
        sections.append(
            "\n".join(current_section)
        )

    return [
        section
        for section in sections
        if section.strip()
    ]


# ============================================================
# CERTIFICATION SECTION EXTRACTION
# ============================================================

def _extract_certification_sections(
    text: str,
) -> List[str]:
    """
    Extract only certification sections.

    This prevents certification regexes from accidentally
    consuming Education, Projects, Experience, etc.
    """

    if not text:
        return []

    lines = text.splitlines()

    sections = []
    current_section = []

    inside_certifications = False

    for line in lines:

        # ----------------------------------------------------
        # Certification section starts
        # ----------------------------------------------------

        if _is_certification_header(line):

            if inside_certifications and current_section:
                sections.append(
                    "\n".join(current_section)
                )

            current_section = []
            inside_certifications = True
            continue

        # ----------------------------------------------------
        # Any other known section starts
        # ----------------------------------------------------

        if (
            _is_experience_header(line)
            or _is_non_experience_header(line)
        ):

            # Do not let "certifications" itself close
            # the section because it was handled above.
            if inside_certifications and current_section:
                sections.append(
                    "\n".join(current_section)
                )

            current_section = []
            inside_certifications = False
            continue

        # ----------------------------------------------------
        # Store certification lines
        # ----------------------------------------------------

        if inside_certifications:
            current_section.append(line)

    # --------------------------------------------------------
    # Save final certification section
    # --------------------------------------------------------

    if inside_certifications and current_section:
        sections.append(
            "\n".join(current_section)
        )

    return [
        section
        for section in sections
        if section.strip()
    ]


# ============================================================
# DATE HELPERS
# ============================================================

MONTHS = {
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


def _month_index(
    year: int,
    month: int,
) -> int:
    """
    Convert year/month into a comparable integer.
    """

    return year * 12 + month


def _parse_month(
    month_name: str,
) -> int:
    """
    Convert month name to month number.
    """

    return MONTHS[
        month_name.lower()
    ]


# ============================================================
# EXPERIENCE DATE EXTRACTION
# ============================================================

def _extract_experience_intervals(
    text: str,
) -> List[Tuple[int, int]]:
    """
    Extract employment date intervals.

    Supported examples:

        Jan 2022 - Dec 2024
        January 2022 - December 2024
        Jan 2022 - Present
        2020 - 2024
        2022 - Present
    """

    if not text:
        return []

    now = datetime.now()

    current_year = now.year
    current_month = now.month

    intervals = []

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

    # ========================================================
    # Month + Year ranges
    # ========================================================

    month_range_pattern = re.compile(
        rf"\b"
        rf"({month_pattern})"
        rf"\s+"
        rf"(\d{{4}})"
        rf"\s*"
        rf"(?:-|–|—|to)"
        rf"\s*"
        rf"(?:"
        rf"({month_pattern})"
        rf"\s+"
        rf"(\d{{4}})"
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

            start_month_name = match.group(1)
            start_year = int(match.group(2))

            end_month_name = match.group(3)
            end_year_value = match.group(4)
            end_present = match.group(5)

            start_month = _parse_month(
                start_month_name
            )

            start_index = _month_index(
                start_year,
                start_month,
            )

            # ------------------------------------------------
            # Present/current/now
            # ------------------------------------------------

            if end_present:

                end_index = _month_index(
                    current_year,
                    current_month,
                )

            # ------------------------------------------------
            # Specific month/year
            # ------------------------------------------------

            else:

                end_month = _parse_month(
                    end_month_name
                )

                end_year = int(
                    end_year_value
                )

                end_index = _month_index(
                    end_year,
                    end_month,
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if (
                1950 <= start_year <= current_year
                and start_index <= end_index
                and end_index <= _month_index(
                    current_year,
                    current_month,
                )
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

    # ========================================================
    # Year-only ranges
    # ========================================================

    year_range_pattern = re.compile(
        r"\b"
        r"(19\d{2}|20\d{2})"
        r"\s*"
        r"(?:-|–|—|to)"
        r"\s*"
        r"(present|current|now|19\d{2}|20\d{2})"
        r"\b",
        re.IGNORECASE,
    )

    for match in year_range_pattern.finditer(
        text.lower()
    ):

        # Ignore a range already captured by the
        # month/year pattern.
        overlapping = False

        for start, end in consumed_spans:

            if (
                match.start() < end
                and match.end() > start
            ):
                overlapping = True
                break

        if overlapping:
            continue

        try:

            start_year = int(
                match.group(1)
            )

            end_value = match.group(2).lower()

            start_index = _month_index(
                start_year,
                1,
            )

            if end_value in {
                "present",
                "current",
                "now",
            }:

                end_index = _month_index(
                    current_year,
                    current_month,
                )

            else:

                end_year = int(
                    end_value
                )

                end_index = _month_index(
                    end_year,
                    1,
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

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

        except (
            ValueError,
            IndexError,
        ):
            continue

    return intervals


# ============================================================
# MERGE EXPERIENCE INTERVALS
# ============================================================

def _merge_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Merge overlapping employment periods.

    Example:

        Job A: 2020 - 2022
        Job B: 2021 - 2023

    Result:

        2020 - 2023

    This prevents double-counting overlapping jobs.
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

        # Overlap or directly adjacent.
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


# ============================================================
# EXPERIENCE CALCULATION
# ============================================================

def _calculate_experience_years(
    intervals: List[Tuple[int, int]],
) -> Optional[float]:
    """
    Calculate total professional experience from
    merged employment intervals.
    """

    if not intervals:
        return None

    merged = _merge_intervals(
        intervals
    )

    total_months = 0

    for start, end in merged:

        duration = end - start

        # Include the ending month.
        if duration >= 0:
            duration += 1

        total_months += duration

    if total_months <= 0:
        return None

    total_years = (
        total_months / 12.0
    )

    if total_years > 50:
        return None

    return round(
        total_years,
        1,
    )


# ============================================================
# MAIN EXPERIENCE FUNCTION
# ============================================================

def extract_experience_years(
    text: str,
) -> Optional[float]:
    """
    Extract total professional experience.

    Priority:

    1. Explicit experience statement.
    2. Employment date ranges inside experience sections.

    Important:
    If there is no explicit experience statement and no
    identifiable experience section, this function returns
    None instead of guessing from random dates.
    """

    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # ========================================================
    # METHOD 1
    # Explicit experience statements
    # ========================================================

    direct_patterns = [

        # Example:
        # 3 years of experience
        # 3+ years experience
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?"
        r"\s*(?:of\s*)?"
        r"(?:professional\s*)?"
        r"(?:experience|exp|work)",

        # Example:
        # Experience: 3 years
        r"(?:experience|exp|work)"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*years?",

        # Example:
        # 3 yrs experience
        r"(\d+(?:\.\d+)?)"
        r"\s*\+?\s*yrs?"
        r"\s*(?:of\s*)?"
        r"(?:professional\s*)?"
        r"(?:experience|exp|work)",
    ]

    for pattern in direct_patterns:

        match = re.search(
            pattern,
            text_lower,
        )

        if match:

            try:

                years = float(
                    match.group(1)
                )

                if 0 <= years <= 50:

                    return round(
                        years,
                        1,
                    )

            except (
                ValueError,
                IndexError,
            ):
                pass

    # ========================================================
    # METHOD 2
    # Employment section date ranges
    # ========================================================

    experience_sections = (
        _extract_experience_sections(text)
    )

    # Never guess from dates elsewhere in the resume.
    if not experience_sections:
        return None

    all_intervals = []

    for section in experience_sections:

        intervals = (
            _extract_experience_intervals(
                section
            )
        )

        all_intervals.extend(
            intervals
        )

    return _calculate_experience_years(
        all_intervals
    )


# ============================================================
# EDUCATION EXTRACTION
# ============================================================

def extract_education(
    text: str,
) -> Dict[str, bool]:
    """
    Detect education levels mentioned in a resume.

    Returns:

        {
            "phd": bool,
            "masters": bool,
            "bachelors": bool,
            "diploma": bool
        }
    """

    education = {
        "phd": False,
        "masters": False,
        "bachelors": False,
        "diploma": False,
    }

    if not text:
        return education

    text_lower = text.lower()

    for degree_type, keywords in (
        EDUCATION_KEYWORDS.items()
    ):

        for keyword in keywords:

            keyword = (
                keyword.lower().strip()
            )

            pattern = (
                r"(?<!\w)"
                + re.escape(keyword)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                text_lower,
            ):

                education[
                    degree_type
                ] = True

                break

    return education


# ============================================================
# CERTIFICATION HELPERS
# ============================================================

def _clean_certification_line(
    line: str,
) -> str:
    """
    Clean a certification line without accidentally
    joining unrelated resume sections.
    """

    line = line.strip()

    # Remove common bullet characters.
    line = re.sub(
        r"^[•●▪◦\-–—*]+\s*",
        "",
        line,
    )

    # Remove excessive whitespace.
    line = re.sub(
        r"\s+",
        " ",
        line,
    ).strip()

    return line


def _looks_like_certification(
    line: str,
) -> bool:
    """
    Determine whether a line looks like a certification.
    """

    line_lower = line.lower()

    certification_indicators = [
        "certified",
        "certification",
        "certificate",
        "associate",
        "professional",
        "credential",
        "license",
    ]

    return any(
        indicator in line_lower
        for indicator in certification_indicators
    )


def _extract_certifications_from_section(
    section: str,
) -> List[str]:
    """
    Extract certification names from a dedicated
    certification section.

    In a dedicated certification section, a line does not
    necessarily need to contain the word 'certified'.

    Example:

        Certifications
        AWS Certified Cloud Practitioner
        Oracle AI Foundations Associate
        Azure Administrator Associate

    All three can be extracted.
    """

    certifications = []

    lines = section.splitlines()

    for line in lines:

        cleaned = _clean_certification_line(
            line
        )

        if not cleaned:
            continue

        # Avoid extremely long descriptive paragraphs.
        if len(cleaned) > 150:
            continue

        # Ignore obvious date-only lines.
        if re.fullmatch(
            r"[\s\d/.\-–—to]+",
            cleaned,
        ):
            continue

        # Ignore common descriptive sentences.
        if (
            len(cleaned.split()) > 15
            and not _looks_like_certification(
                cleaned
            )
        ):
            continue

        # Ignore obvious section headings.
        if (
            _normalize_header(cleaned)
            in NON_EXPERIENCE_HEADERS
            or _normalize_header(cleaned)
            in EXPERIENCE_HEADERS
        ):
            continue

        # ----------------------------------------------------
        # A dedicated certification section is strong
        # evidence that each reasonable line is a certificate.
        # ----------------------------------------------------

        if cleaned not in certifications:

            certifications.append(
                cleaned
            )

    return certifications


# ============================================================
# FALLBACK CERTIFICATION EXTRACTION
# ============================================================

def _extract_certifications_without_section(
    text: str,
) -> List[str]:
    """
    Fallback for resumes that do not have a dedicated
    Certifications heading.

    Only lines that explicitly contain certification-related
    words are considered.

    This is intentionally conservative.
    """

    if not text:
        return []

    certifications = []

    lines = text.splitlines()

    for line in lines:

        cleaned = _clean_certification_line(
            line
        )

        if not cleaned:
            continue

        if len(cleaned) > 150:
            continue

        if not _looks_like_certification(
            cleaned
        ):
            continue

        # ----------------------------------------------------
        # Ignore lines that are actually section headings.
        # ----------------------------------------------------

        normalized = _normalize_header(
            cleaned
        )

        if (
            normalized in NON_EXPERIENCE_HEADERS
            or normalized in EXPERIENCE_HEADERS
        ):
            continue

        # ----------------------------------------------------
        # Avoid capturing huge sentences.
        # ----------------------------------------------------

        if len(cleaned.split()) > 15:
            continue

        if cleaned not in certifications:

            certifications.append(
                cleaned
            )

    return certifications


# ============================================================
# CERTIFICATION EXTRACTION
# ============================================================

def extract_certifications(
    text: str,
) -> List[str]:
    """
    Extract certifications from resume text.

    Strategy:

    1. Prefer dedicated Certifications sections.
    2. If no certification section exists, use a conservative
       line-based fallback.
    3. Never run broad certification regexes across the entire
       resume because that can capture Education/Projects/etc.
    """

    if not text:
        return []

    certifications = []

    # ========================================================
    # METHOD 1
    # Dedicated certification sections
    # ========================================================

    certification_sections = (
        _extract_certification_sections(
            text
        )
    )

    for section in certification_sections:

        section_certs = (
            _extract_certifications_from_section(
                section
            )
        )

        for cert in section_certs:

            if cert not in certifications:

                certifications.append(
                    cert
                )

    # ========================================================
    # METHOD 2
    # Fallback when no dedicated section exists
    # ========================================================

    if not certification_sections:

        fallback_certs = (
            _extract_certifications_without_section(
                text
            )
        )

        for cert in fallback_certs:

            if cert not in certifications:

                certifications.append(
                    cert
                )

    # ========================================================
    # Final cleanup
    # ========================================================

    cleaned_certifications = []

    for cert in certifications:

        cert = re.sub(
            r"\s+",
            " ",
            cert,
        ).strip()

        if (
            cert
            and cert not in cleaned_certifications
        ):

            cleaned_certifications.append(
                cert
            )

    return cleaned_certifications[:10]


# ============================================================
# JOB ROLE EXTRACTION
# ============================================================

def extract_recent_roles(
    text: str,
    top_n: int = 5,
) -> List[str]:
    """
    Extract possible job titles from resume text.

    Detects titles such as:

        Software Engineer
        Machine Learning Engineer
        Data Scientist
        Backend Developer
        Data Analyst
        Project Manager
    """

    if not text:
        return []

    roles = []

    title_keywords = [
        "engineer",
        "developer",
        "analyst",
        "manager",
        "lead",
        "architect",
        "scientist",
        "specialist",
        "consultant",
        "director",
        "executive",
        "designer",
        "administrator",
        "coordinator",
        "intern",
        "trainee",
    ]

    lines = text.splitlines()

    for line in lines:

        line_clean = line.strip()

        if not line_clean:
            continue

        if (
            len(line_clean) < 5
            or len(line_clean) > 100
        ):
            continue

        line_lower = line_clean.lower()

        # ----------------------------------------------------
        # Ignore contact information
        # ----------------------------------------------------

        if re.search(
            r"@|phone|email|linkedin|github|http|www\.",
            line_lower,
        ):
            continue

        # ----------------------------------------------------
        # Ignore lines containing only dates
        # ----------------------------------------------------

        if re.fullmatch(
            r"[\s\d/.\-–—to]+",
            line_clean,
        ):
            continue

        # ----------------------------------------------------
        # Detect title keywords
        # ----------------------------------------------------

        has_title_keyword = any(
            re.search(
                rf"\b{re.escape(keyword)}\b",
                line_lower,
            )
            for keyword in title_keywords
        )

        has_role_prefix = re.match(
            r"^(position|role|title|designation|job title)"
            r"\s*:?\s*",
            line_lower,
        )

        if (
            has_title_keyword
            or has_role_prefix
        ):

            cleaned = re.sub(
                r"^(position|role|title|designation|job title)"
                r"\s*:?\s*",
                "",
                line_lower,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"\s+",
                " ",
                cleaned,
            ).strip()

            # Avoid full sentences.
            if len(cleaned.split()) > 10:
                continue

            if (
                cleaned
                and cleaned not in roles
            ):

                roles.append(
                    cleaned
                )

    return roles[:top_n]


# ============================================================
# ALL FEATURES
# ============================================================

def extract_all_resume_features(
    text: str,
) -> Dict:
    """
    Extract all structured resume features.
    """

    return {
        "experience_years": extract_experience_years(
            text
        ),
        "education": extract_education(
            text
        ),
        "certifications": extract_certifications(
            text
        ),
        "recent_roles": extract_recent_roles(
            text
        ),
    }