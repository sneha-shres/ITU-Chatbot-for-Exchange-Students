#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse LearnIT (Moodle) course HTML files and extract structured fields.

- Inputs: a folder with .html files (downloaded course pages)
- Outputs: JSON and CSV files with extracted fields
- Parsing: BeautifulSoup (bs4), robust to minor structural differences

Set INPUT_DIR, OUT_JSON, and OUT_CSV below.
"""

import copy
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------
# Configuration: set your paths here
# ---------------------------------------
INPUT_DIR = "Courses/course_pages"           
OUT_JSON = "Courses/output/courses.json"      
OUT_CSV  = "Courses/output/courses.csv"      

# ---------------------------
# Helpers: text normalization
# ---------------------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def text_of(node: Tag) -> str:
    return norm_space(node.get_text(" ", strip=True)) if node else ""

def is_heading_tag(tag: Tag) -> bool:
    # Treat only real headings or role=heading as section boundaries; NOT <strong>
    if not isinstance(tag, Tag):
        return False
    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True
    if tag.has_attr("role") and str(tag.get("role")).lower() == "heading":
        return True
    # Accept <dt> as a “title” in definition lists in some exports
    if tag.name == "dt":
        return True
    return False

def is_heading_like(tag: Tag) -> bool:
    # Used only for locating section titles; allow strong as title text
    if not isinstance(tag, Tag):
        return False
    if is_heading_tag(tag):
        return True
    if tag.name in {"strong"}:
        return True
    return False

def make_label_regex(labels: Iterable[str]) -> re.Pattern:
    escaped = [re.escape(l) for l in labels]
    pattern = r"(^|\b|\s)(" + "|".join(escaped) + r")(\b|:|\s|$)"
    return re.compile(pattern, flags=re.IGNORECASE)

# ---------------------------------------
# Section extraction by heading label(s)
# ---------------------------------------

def soup_new_fragment():
    frag = BeautifulSoup("", "html.parser")
    wrapper = frag.new_tag("div")
    frag.append(wrapper)
    return frag, wrapper

def heading_level(tag: Tag) -> int:
    if tag.name in {"h1","h2","h3","h4","h5","h6"}:
        return int(tag.name[1])
    if tag.name == "dt":
        return 4
    # strong is treated as title-like but not a boundary when collecting
    return 6

def find_heading_tag(soup: BeautifulSoup, labels: Iterable[str]) -> Optional[Tag]:
    label_list = list(labels)
    label_re = make_label_regex(label_list)

    def normalize_label(s: str) -> str:
        s = norm_space(s)
        s = re.sub(r":\s*$", "", s)  # drop trailing colon
        return s.lower()

    wanted_exact = set(normalize_label(l) for l in label_list)

    # Pass 1: prefer exact label match (case-insensitive, ignoring trailing colon)
    for tag in soup.find_all(is_heading_like):
        t = text_of(tag)
        if normalize_label(t) in wanted_exact:
            return tag

    # Pass 2: fallback to substring/regex match
    for tag in soup.find_all(is_heading_like):
        t = text_of(tag)
        if label_re.search(t):
            return tag
    return None

def collect_section_after_heading(heading_tag: Tag) -> Tag:
    """
    Return a <div> containing all siblings after heading_tag until the next
    real heading-like tag (h1–h6, role=heading, dt) of same/higher level.
    NOTE: strong tags do NOT terminate the section (fixes missing bullet lists).
    This handles cases where headings are nested inside p or div tags.
    """
    frag, container = soup_new_fragment()
    start_level = heading_level(heading_tag)
    
    # Convert next_siblings to a list FIRST to avoid iterator issues when modifying DOM
    siblings = list(heading_tag.next_siblings)
    
    # Collect next siblings of the heading
    for sib in siblings:
        if isinstance(sib, NavigableString):
            t = norm_space(str(sib))
            if t:
                p = frag.new_tag("p")
                p.string = t
                container.append(p)
            continue
        if isinstance(sib, Tag):
            # If the starting heading is a <strong>, treat the next <strong> in the same parent as boundary.
            if heading_tag.name == "strong" and sib.name == "strong":
                break
            if is_heading_tag(sib) and heading_level(sib) <= start_level:
                break
            # Copy the tag to avoid modifying the original DOM
            container.append(copy.copy(sib))
    
    return container

def extract_paragraphs_text(container: Tag) -> str:
    # Include p/div/dd for exports that use definition lists for values
    parts: List[str] = []
    for node in container.descendants:
        if isinstance(node, Tag) and node.name in {"p", "div", "dd"}:
            t = norm_space(node.get_text(" ", strip=True))
            if t:
                parts.append(t)
    if not parts:
        t = norm_space(container.get_text(" ", strip=True))
        if t:
            parts.append(t)
    return norm_space(" ".join(parts))

def extract_list_items(container: Tag) -> List[str]:
    items: List[str] = []
    for ul in container.find_all(["ul", "ol"]):
        for li in ul.find_all("li"):
            t = norm_space(li.get_text(" ", strip=True))
            if t:
                items.append(t)
    return items

# ---------------------------------------
# Field-specific extractors
# ---------------------------------------

def extract_title(soup: BeautifulSoup) -> Optional[str]:
    for tag_name in ["h1", "h2"]:
        h = soup.find(tag_name)
        if h and norm_space(h.get_text()):
            return clean_title(norm_space(h.get_text()))
    if soup.title and soup.title.string:
        return clean_title(norm_space(soup.title.string))
    return None

def clean_title(t: str) -> str:
    # Remove trailing site label
    t = re.sub(r"\s*[-–|]\s*learnit.*$", "", t, flags=re.IGNORECASE)
    return t

def extract_semester_block(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    From the 'Course semester' section, parse Semester, Start, End when present.
    Also looks for semester dates anywhere on the page in row format.
    """
    section = extract_section_container(soup, ["Course semester", "COURSE SEMESTER"])
    result = {"semester_label": None, "start": None, "end": None}
    
    # First try traditional section extraction
    if section:
        # Prefer dt/dd or th/td pairs
        kv = extract_kv_from_section(section)
        # Keys may appear as: Semester, Start, End (or localized)
        for k, v in kv.items():
            kl = k.lower()
            if "semester" in kl:
                result["semester_label"] = v
            elif "start" in kl:
                result["start"] = v
            elif "end" in kl:
                result["end"] = v

        # Also support row-based structure within the section
        for row in section.find_all("div", class_="row"):
            cols = row.find_all("div", class_="col-md-12")
            if len(cols) >= 2:
                label_text = text_of(cols[0]).lower()
                value_text = text_of(cols[1])
                if "semester" in label_text and not result["semester_label"]:
                    result["semester_label"] = value_text
                elif label_text.strip().startswith("start") and not result["start"]:
                    result["start"] = value_text
                elif label_text.strip().startswith("end") and not result["end"]:
                    result["end"] = value_text

        # Fallbacks via regex in full section text
        text = section.get_text(" ", strip=True)
        if not result["semester_label"]:
            m = re.search(r"\b(Semester)\b[:\s]*([A-Za-zÆØÅæøå]+(?:\s+\d{4})?)", text)
            if m:
                result["semester_label"] = norm_space(m.group(2))
        if not result["start"]:
            m = re.search(r"\bStart\b[:\s]*([0-9]{1,2}\s+[A-Za-z]+\s+\d{4})", text, flags=re.IGNORECASE)
            if m:
                result["start"] = m.group(1)
        if not result["end"]:
            m = re.search(r"\bEnd\b[:\s]*([0-9]{1,2}\s+[A-Za-z]+\s+\d{4})", text, flags=re.IGNORECASE)
            if m:
                result["end"] = m.group(1)
    
    # Global fallback: look for semester dates anywhere on page in row format
    if not result["start"] or not result["end"]:
        for row in soup.find_all("div", class_="row"):
            cols = row.find_all("div", class_="col-md-12")
            if len(cols) >= 2:
                label_text = text_of(cols[0]).strip().lower()
                value_text = text_of(cols[1]).strip()
                
                # Look for "Start" and "End" labels
                if label_text == "start" and not result["start"]:
                    result["start"] = value_text
                elif label_text == "end" and not result["end"]:
                    result["end"] = value_text
                
                # Also look for semester label (like "Efterår 2026")
                if not result["semester_label"] and any(season in label_text for season in ["efterår", "forår", "vinter", "sommer", "autumn", "spring", "winter", "summer"]):
                    result["semester_label"] = value_text

    return result

def extract_semester(soup: BeautifulSoup, title_text: Optional[str]) -> Optional[str]:
    sb = extract_semester_block(soup)
    if sb["semester_label"]:
        return sb["semester_label"]
    # Infer from title or body
    t = title_text or ""
    m = re.search(r"\b(Spring|Summer|Autumn|Fall|Winter|Efterår|Forår|Vinter|Sommer)\b\s*(\d{4})\b", t, flags=re.IGNORECASE)
    if m:
        season = m.group(1)
        year = m.group(2)
        return f"{season} {year}"
    body = soup.get_text(" ", strip=True)
    m2 = re.search(r"\b(Spring|Summer|Autumn|Fall|Winter|Efterår|Forår|Vinter|Sommer)\b\s*(\d{4})\b", body, flags=re.IGNORECASE)
    if m2:
        season = m2.group(1)
        year = m2.group(2)
        return f"{season} {year}"
    return None

def extract_course_code(soup: BeautifulSoup, title_text: Optional[str]) -> Optional[str]:
    # Try inside Course info first
    info = extract_course_info_map(soup)
    if info.get("course_code"):
        return info["course_code"]
    
    # Heuristic fallback anywhere on page
    t = " " + (title_text or "") + " " + soup.get_text(" ", strip=True) + " "
    m = re.search(r"\b([A-Z]{2,8}[A-Z]?\d{2,4}[A-Z]*)\b", t)
    if m:
        return m.group(1)
    
    # Additional pattern for course codes like "KSADALG1KU"
    m2 = re.search(r"\b([A-Z]{6,12})\b", t)
    if m2 and len(m2.group(1)) >= 6:  # Longer codes are more likely to be course codes
        return m2.group(1)
    
    return None

def extract_description(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    # Abstract and Description sections
    abstract = extract_section_text(soup, ["Abstract", "ABSTRACT"])
    desc = extract_section_text(soup, ["Description", "DESCRIPTION"])
    return (abstract or None, desc or None)

def extract_learning_outcomes(soup: BeautifulSoup) -> List[str]:
    section = extract_section_container(
        soup,
        [
            "Intended learning outcomes",
            "INTENDED LEARNING OUTCOMES",
            "Learning outcomes",
            "LEARNING OUTCOMES",
            "Intended outcomes",
            "Objectives",
        ],
    )
    if section:
        items = extract_list_items(section)
        if items:
            return items
        txt = extract_paragraphs_text(section)
        if txt:
            parts = [p.strip() for p in re.split(r"[;\n]+", txt) if p.strip()]
            # Remove the intro line if present
            parts = [p for p in parts if not p.lower().startswith("after the course")]
            return parts
    return []

def extract_teachers(soup: BeautifulSoup) -> List[str]:
    section = extract_section_container(
        soup,
        [
            "Staff",
            "STAFF",
            "Teachers",
            "Teacher",
            "Course responsible",
            "Course manager",
            "Lecturer",
            "Instructors",
        ],
    )
    if not section:
        return []
    
    names: List[str] = []
    
    # Look for names in links
    for a in section.find_all("a"):
        t = norm_space(a.get_text())
        if looks_like_name(t):
            names.append(t)
    
    # Look for names in text after Staff heading
    text = extract_paragraphs_text(section)
    for token in re.split(r"[,\n;•]+", text):
        t = norm_space(token)
        if looks_like_name(t):
            names.append(t)
    
    # Clean up names (remove emails, etc.)
    names = [re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", n).strip() for n in names]
    
    # Deduplicate and validate
    cleaned, seen = [], set()
    for n in names:
        if not n or len(n) < 3:
            continue
        if re.search(r"[A-Za-zÆØÅæøå]", n) and " " in n:
            key = re.sub(r"\s+", " ", n).strip().lower()
            if key not in seen:
                cleaned.append(re.sub(r"\s+", " ", n).strip())
                seen.add(key)
    
    return cleaned

def looks_like_name(s: str) -> bool:
    words = s.split()
    if not (2 <= len(words) <= 5):
        return False
    if sum(1 for w in words if w[:1].isupper()) < 2:
        return False
    if s.lower() in {"teachers", "staff", "contact", "exam", "programme"}:
        return False
    return True

def extract_ects(soup: BeautifulSoup) -> Optional[str]:
    info = extract_course_info_map(soup)
    if info.get("ects"):
        return info["ects"]
    
    # Fallback global scan
    body = soup.get_text(" ", strip=True)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*ECTS\b", body, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace(",", ".")
    
    # Additional fallback for "ECTS points"
    m2 = re.search(r"ECTS\s+points?[:\s]*(\d+(?:[.,]\d+)?)", body, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).replace(",", ".")
    
    return None

# ---------------------------------------
# Key–value parsing utilities
# ---------------------------------------

def extract_kv_from_section(section: Tag) -> Dict[str, str]:
    """
    Extract label->value pairs from a section that may use:
    - <dl><dt>Label</dt><dd>Value</dd>
    - <table><tr><th>Label</th><td>Value</td></tr></table>
    - <strong>Label:</strong> Value
    - Plain text lines like "Label: Value"
    """
    kv: Dict[str, str] = {}

    # dt/dd
    for dt in section.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            k = text_of(dt)
            v = text_of(dd)
            if k and v:
                kv[k] = v

    # th/td
    for row in section.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and td:
            k = text_of(th)
            v = text_of(td)
            if k and v:
                kv[k] = v

    # strong label followed by siblings
    for strong in section.find_all("strong"):
        k = re.sub(r":\s*$", "", text_of(strong))
        # Build value from following siblings within the same parent, until another strong
        parts = []
        for sib in strong.next_siblings:
            if isinstance(sib, NavigableString):
                t = norm_space(str(sib))
                if t:
                    parts.append(t)
            elif isinstance(sib, Tag):
                if sib.name == "strong":
                    break
                t = text_of(sib)
                if t:
                    parts.append(t)
        v = norm_space(" ".join(parts))
        if k and v:
            kv[k] = v

    # Plain text lines with colon
    text = section.get_text("\n", strip=True)
    for line in [l.strip() for l in text.split("\n") if ":" in l]:
        k, _, v = line.partition(":")
        k = norm_space(k)
        v = norm_space(v)
        if k and v and k not in kv:
            kv[k] = v

    return kv

def extract_course_info_map(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    Normalize common fields from the 'Course info' section.
    Handles both traditional sections and row-based layouts.
    """
    out = {
        "language": None,
        "ects": None,
        "course_code": None,
        "participants_max": None,
        "location": None,
        "campus": None,
        "level": None,
        "course_type": None,
        "department": None,
        "programme_name": None,
        "offered_guest": None,
        "offered_exchange": None,
        "offered_single_subject": None,
        "price_eu_eea": None,
    }
    
    # First try traditional section extraction
    section = extract_section_container(soup, ["Course info", "COURSE INFO"])
    if section:
        kv = extract_kv_from_section(section)
        for k, v in kv.items():
            kl = k.lower()
            if "language" in kl:
                out["language"] = v
            elif "ects" in kl:
                m = re.search(r"(\d+(?:[.,]\d+)?)", v)
                out["ects"] = m.group(1).replace(",", ".") if m else v
            elif "course code" in kl or kl == "coursecode":
                out["course_code"] = v
            elif "participants" in kl and ("max" in kl or "maximum" in kl):
                out["participants_max"] = v
            elif "location" in kl or "room" in kl:
                out["location"] = v
            elif "campus" in kl:
                out["campus"] = v
            elif "level" in kl:
                out["level"] = v
            elif "type" in kl and "course" in kl:
                out["course_type"] = v
            elif "department" in kl or "institute" in kl:
                out["department"] = v
            elif "programme" in kl or "program" in kl:
                out["programme_name"] = v
            elif "offered to guest students" in kl:
                out["offered_guest"] = v
            elif "offered to exchange students" in kl:
                out["offered_exchange"] = v
            elif "offered as a single subject" in kl:
                out["offered_single_subject"] = v
            elif "price" in kl and ("eu" in kl or "eea" in kl):
                out["price_eu_eea"] = v
    
    # Fallback: look for row-based structure (col-md-12 pattern)
    for row in soup.find_all("div", class_="row"):
        cols = row.find_all("div", class_="col-md-12")
        if len(cols) >= 2:
            label_text = text_of(cols[0])
            value_text = text_of(cols[1])
            if not label_text or not value_text:
                continue
                
            kl = label_text.lower().strip()
            if "language" in kl:
                out["language"] = value_text.strip()
            elif "ects" in kl:
                m = re.search(r"(\d+(?:[.,]\d+)?)", value_text)
                out["ects"] = m.group(1).replace(",", ".") if m else value_text.strip()
            elif "course code" in kl:
                out["course_code"] = value_text.strip()
            elif "participants" in kl and ("max" in kl or "maximum" in kl):
                out["participants_max"] = value_text.strip()
            elif "location" in kl or "room" in kl:
                out["location"] = value_text.strip()
            elif "campus" in kl:
                out["campus"] = value_text.strip()
            elif "level" in kl:
                out["level"] = value_text.strip()
            elif "type" in kl and "course" in kl:
                out["course_type"] = value_text.strip()
            elif "department" in kl or "institute" in kl:
                out["department"] = value_text.strip()
            elif "programme" in kl or "program" in kl:
                out["programme_name"] = value_text.strip()
            elif "offered to guest students" in kl:
                out["offered_guest"] = value_text.strip()
            elif "offered to exchange students" in kl:
                out["offered_exchange"] = value_text.strip()
            elif "offered as a single subject" in kl:
                out["offered_single_subject"] = value_text.strip()
            elif "price" in kl and ("eu" in kl or "eea" in kl):
                out["price_eu_eea"] = value_text.strip()

    return out

def extract_labeled_value(soup: BeautifulSoup, labels: Iterable[str]) -> Optional[str]:
    """
    Page-wide labeled value finder (rarely needed now that sections are parsed).
    """
    label_re = make_label_regex(labels)

    for dt in soup.find_all("dt"):
        if label_re.search(text_of(dt)):
            dd = dt.find_next_sibling("dd")
            if dd:
                return text_of(dd)

    for th in soup.find_all("th"):
        if label_re.search(text_of(th)):
            td = th.find_next_sibling("td")
            if td:
                return text_of(td)

    for strong in soup.find_all("strong"):
        if label_re.search(text_of(strong)):
            parts = []
            for sib in strong.next_siblings:
                if isinstance(sib, NavigableString):
                    t = norm_space(str(sib))
                    if t:
                        parts.append(t)
                elif isinstance(sib, Tag):
                    if is_heading_tag(sib):
                        break
                    t = text_of(sib)
                    if t:
                        parts.append(t)
            if parts:
                return norm_space(" ".join(parts))

    return None

def extract_section_text(soup: BeautifulSoup, labels: Iterable[str]) -> Optional[str]:
    container = extract_section_container(soup, labels)
    if not container:
        return None
    return extract_paragraphs_text(container)

def extract_section_container(soup: BeautifulSoup, labels: Iterable[str]) -> Optional[Tag]:
    heading = find_heading_tag(soup, labels)
    if not heading:
        return None
    return collect_section_after_heading(heading)

# ---------------------------------------
# Rich section helpers and broader scraping
# ---------------------------------------

def normalize_heading_label(label: str) -> str:
    label = norm_space(label).lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "section"

def extract_section_rich(soup: BeautifulSoup, labels: Iterable[str]) -> Optional[Dict]:
    container = extract_section_container(soup, labels)
    if not container:
        return None
    return {
        "text": extract_paragraphs_text(container) or None,
        "items": extract_list_items(container) or [],
        "kv": extract_kv_from_section(container) or {},
    }

def extract_prerequisites(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    req = extract_section_rich(
        soup,
        [
            "Prerequisites",
            "PREREQUISITES",
            "Requirements",
            "REQUIREMENTS",
            "Formal prerequisites",
            "Formal requirements",
        ],
    )
    rec = extract_section_rich(
        soup,
        [
            "Recommended prerequisites",
            "Recommended",
            "RECOMMENDED PREREQUISITES",
            "Recommended background",
        ],
    )
    required_text = (req or {}).get("text") if req else None
    recommended_text = (rec or {}).get("text") if rec else None
    return (required_text or None), (recommended_text or None)

def extract_content_materials(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    content = extract_section_rich(
        soup,
        [
            "Content",
            "Course content",
            "CONTENT",
            "Course topics",
            "Topics",
            "Syllabus",
        ],
    )
    materials = extract_section_rich(
        soup,
        [
            "Materials",
            "Course material",
            "Materials for the course",
            "Course literature",
            "Books",
        ],
    )
    literature = extract_section_rich(
        soup,
        ["Literature", "Reading list", "LITERATURE", "READING LIST", "References"],
    )
    content_text = (content or {}).get("text") or ("; ".join((content or {}).get("items", [])) if content else None)
    materials_text = (materials or {}).get("text") or ("; ".join((materials or {}).get("items", [])) if materials else None)
    literature_text = (literature or {}).get("text") or ("; ".join((literature or {}).get("items", [])) if literature else None)
    return (content_text or None), (materials_text or None), (literature_text or None)

def extract_assessment(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    rich = extract_section_rich(
        soup,
        [
            "Assessment",
            "ASSESSMENT",
            "Exam",
            "EXAM",
            "Evaluation",
            "EVALUATION",
            "Grading",
            "Exam format",
            "Examination",
            "Ordinary exam",
            "ORDINARY EXAM",
        ],
    )
    if not rich:
        rich = {"text": None, "items": [], "kv": {}}

    # Extract exam type and variation from the section text
    exam_type = None
    exam_variation = None
    
    # Look for exam info in the section text
    if rich.get("text"):
        text = rich["text"]
        # Look for "Exam type:" followed by the type (stop before "Exam variation" or end)
        exam_type_match = re.search(r"Exam type:\s*([^\n]+?)(?=\s*Exam variation:|$)", text, re.IGNORECASE)
        if exam_type_match:
            exam_type = norm_space(exam_type_match.group(1))
        
        # Look for "Exam variation:" followed by the variation
        exam_variation_match = re.search(r"Exam variation:\s*([^\n]+)", text, re.IGNORECASE)
        if exam_variation_match:
            exam_variation = norm_space(exam_variation_match.group(1))
    
    # Also check the full page text for exam info (fallback)
    if not exam_type or not exam_variation:
        full_text = soup.get_text(" ", strip=True)
        exam_type_match = re.search(r"Exam type:\s*([^\.]+?)(?=\s*Exam variation:|$)", full_text, re.IGNORECASE)
        if exam_type_match:
            exam_type = norm_space(exam_type_match.group(1))
        
        exam_variation_match = re.search(r"Exam variation:\s*([^\.]+)", full_text, re.IGNORECASE)
        if exam_variation_match:
            exam_variation = norm_space(exam_variation_match.group(1))
    
    # Final fallback: extract from prerequisites text if it contains exam info
    if not exam_type or not exam_variation:
        prereq_section = extract_section_container(soup, ["Formal prerequisites", "Prerequisites"])
        if prereq_section:
            prereq_text = extract_paragraphs_text(prereq_section)
            if prereq_text:
                exam_type_match = re.search(r"Exam type:\s*([^\.]+?)(?=\s*Exam variation:|$)", prereq_text, re.IGNORECASE)
                if exam_type_match:
                    exam_type = norm_space(exam_type_match.group(1))
                
                exam_variation_match = re.search(r"Exam variation:\s*([^\.]+)", prereq_text, re.IGNORECASE)
                if exam_variation_match:
                    exam_variation = norm_space(exam_variation_match.group(1))
    
    # Fallback: look for strong tags with exam info anywhere on page
    if not exam_type or not exam_variation:
        for strong in soup.find_all("strong"):
            text = text_of(strong).lower()
            if "exam type" in text:
                # Get the text after this strong tag
                next_text = ""
                for sib in strong.next_siblings:
                    if isinstance(sib, NavigableString):
                        text_chunk = str(sib)
                        # Stop if we encounter "Exam variation:" in the text
                        if "exam variation" in text_chunk.lower():
                            break
                        next_text += text_chunk
                    elif isinstance(sib, Tag) and sib.name == "br":
                        next_text += " "
                    elif isinstance(sib, Tag) and sib.name == "strong":
                        break
                    else:
                        break
                if next_text.strip() and not exam_type:
                    exam_type = norm_space(next_text)
            elif "exam variation" in text:
                # Get the text after this strong tag
                next_text = ""
                for sib in strong.next_siblings:
                    if isinstance(sib, NavigableString):
                        next_text += str(sib)
                    elif isinstance(sib, Tag) and sib.name == "br":
                        next_text += " "
                    elif isinstance(sib, Tag) and sib.name == "strong":
                        break
                    else:
                        break
                if next_text.strip() and not exam_variation:
                    exam_variation = norm_space(next_text)

    return {
        "assessment_text": rich.get("text"),
        "assessment_items": rich.get("items", []),
        "assessment_kv": rich.get("kv", {}),
        "exam_type": exam_type,
        "exam_variation": exam_variation,
    }

def extract_links_and_emails(soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    links: List[str] = []
    emails: List[str] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("mailto:"):
            addr = href.split(":", 1)[1]
            if addr and addr not in emails:
                emails.append(addr)
            continue
        if href.startswith("#"):
            continue
        if href not in links:
            links.append(href)
    body = soup.get_text(" ", strip=True)
    for m in re.findall(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}", body):
        if m not in emails:
            emails.append(m)
    return links, emails

def extract_all_sections_map(soup: BeautifulSoup) -> Dict[str, Dict]:
    sections: Dict[str, Dict] = {}
    for tag in soup.find_all(is_heading_like):
        label = text_of(tag)
        if not label:
            continue
        key = normalize_heading_label(label)
        container = collect_section_after_heading(tag)
        entry = {
            "label": label,
            "text": extract_paragraphs_text(container) or None,
            "items": extract_list_items(container) or [],
            "kv": extract_kv_from_section(container) or {},
        }
        # Combine duplicates by appending an index
        if key in sections:
            idx = 2
            while f"{key}_{idx}" in sections:
                idx += 1
            sections[f"{key}_{idx}"] = entry
        else:
            sections[key] = entry
    return sections

# ---------------------------------------
# Main parsing for a single file
# ---------------------------------------

def parse_course_html(html: str, source_path: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    semester_label = extract_semester(soup, title)
    sem_block = extract_semester_block(soup)
    course_info = extract_course_info_map(soup)

    course_code = extract_course_code(soup, title)
    abstract, description = extract_description(soup)
    outcomes = extract_learning_outcomes(soup)
    teachers = extract_teachers(soup)
    ects = extract_ects(soup) or course_info.get("ects")
    prereq_required, prereq_recommended = extract_prerequisites(soup)
    content_text, materials_text, literature_text = extract_content_materials(soup)
    assessment = extract_assessment(soup)
    
    # Extract exam info from prerequisites text if not found in assessment
    if not assessment.get("exam_type") and prereq_required:
        exam_type_match = re.search(r"Exam type:\s*([^E]+?)(?=\s*Exam variation|$)", prereq_required, re.IGNORECASE | re.DOTALL)
        if exam_type_match:
            assessment["exam_type"] = norm_space(exam_type_match.group(1))
    
    if not assessment.get("exam_variation") and prereq_required:
        exam_variation_match = re.search(r"Exam variation:\s*([^\.]+)", prereq_required, re.IGNORECASE)
        if exam_variation_match:
            assessment["exam_variation"] = norm_space(exam_variation_match.group(1))
    
    links, emails = extract_links_and_emails(soup)
    all_sections = extract_all_sections_map(soup)

    # Optional schedule/programme text fallback
    schedule_text = extract_section_text(soup, ["Programme", "PROGRAMME", "Schedule", "Course info", "COURSE INFO"])

    return {
        "source_file": source_path,
        "course_title": title,
        "course_code": course_code,
        "abstract": abstract,
        "description": description,
        "teachers": teachers,                  # list
        "ects": ects,
        "learning_outcomes": outcomes,         # list
        "semester": semester_label,
        "semester_start": sem_block.get("start"),
        "semester_end": sem_block.get("end"),
        "language": course_info.get("language"),
        "participants_max": course_info.get("participants_max"),
        "location": course_info.get("location"),
        "campus": course_info.get("campus"),
        "level": course_info.get("level"),
        "course_type": course_info.get("course_type"),
        "department": course_info.get("department"),
        "programme_name": course_info.get("programme_name"),
        "schedule_or_programme": schedule_text,
        "formal_prerequisites": prereq_required,
        "prerequisites_recommended": prereq_recommended,
        "content": content_text,
        "materials": materials_text,
        "literature": literature_text,
        "assessment_text": assessment.get("assessment_text"),
        "assessment_items": assessment.get("assessment_items", []),
        "assessment_kv": assessment.get("assessment_kv", {}),
        "exam_type": assessment.get("exam_type"),
        "exam_variation": assessment.get("exam_variation"),
        "offered_guest": course_info.get("offered_guest"),
        "offered_exchange": course_info.get("offered_exchange"),
        "offered_single_subject": course_info.get("offered_single_subject"),
        "price_eu_eea": course_info.get("price_eu_eea"),
        "links": links,
        "emails": emails,
        "all_sections": all_sections,
    }

# ---------------------------------------
# I/O utilities and runner
# ---------------------------------------

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def write_json(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def write_csv(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat_records = []
    for r in records:
        r2 = dict(r)
        r2["teachers"] = "; ".join(r.get("teachers") or [])
        r2["learning_outcomes"] = " | ".join(r.get("learning_outcomes") or [])
        r2["assessment_items"] = " | ".join(r.get("assessment_items") or [])
        r2["links"] = " | ".join(r.get("links") or [])
        r2["emails"] = " | ".join(r.get("emails") or [])
        # Drop very large fields from CSV to keep it readable
        r2.pop("assessment_kv", None)
        r2.pop("all_sections", None)
        flat_records.append(r2)
    fieldnames = [
        "source_file",
        "course_title",
        "course_code",
        "abstract",
        "description",
        "teachers",
        "ects",
        "learning_outcomes",
        "semester",
        "semester_start",
        "semester_end",
        "language",
        "participants_max",
        "location",
        "campus",
        "level",
        "course_type",
        "department",
        "programme_name",
        "schedule_or_programme",
        "formal_prerequisites",
        "prerequisites_recommended",
        "content",
        "materials",
        "literature",
        "assessment_text",
        "assessment_items",
        "exam_type",
        "exam_variation",
        "offered_guest",
        "offered_exchange",
        "offered_single_subject",
        "price_eu_eea",
        "links",
        "emails",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in flat_records:
            w.writerow(row)

def run():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    records: List[Dict] = []
    for path in sorted(in_dir.glob("*.html")):
        html = read_text(path)
        record = parse_course_html(html, str(path))
        records.append(record)

    if OUT_JSON:
        write_json(records, Path(OUT_JSON))
    if OUT_CSV:
        write_csv(records, Path(OUT_CSV))

if __name__ == "__main__":
    run()
