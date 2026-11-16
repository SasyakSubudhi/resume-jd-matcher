"""
extract.py - Structured-output extraction module (Deliverable 1).

Converts a raw resume / job description into the fixed JSON schema in
schema.py.

Pipeline (LLM mode):
    raw text -> ChatPromptTemplate (instruction + format instructions from
    PydanticOutputParser) -> Groq LLM -> parse into Pydantic model
    -> if parsing fails: REPAIR prompt (send the error back to the LLM, max 2 tries)
    -> if still failing: offline rule-based extractor.

Offline mode: regex + skill-taxonomy rules, so the app always works.
"""
import json
import re
from pathlib import Path
from typing import Tuple, Type

from pydantic import BaseModel, ValidationError

from .llm import extract_json, get_llm
from .schema import Experience, JobProfile, ResumeProfile
from .skills import find_skills_in_text, normalize_skills

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ----------------------------------------------------------------------------
# Prompts (instruction | context | input data | output indicator)
# ----------------------------------------------------------------------------
RESUME_PROMPT = """You are an expert HR data-extraction assistant.
Task: extract structured information from the RESUME below.
Rules:
- Use only information present in the resume; do not invent anything.
- List every technical skill, tool, language and framework as a separate short item.
- total_experience_years = sum of professional work experience (internships count as their duration).
- Write a neutral 2-line summary.

RESUME:
<<<
{text}
>>>

{format_instructions}
Return ONLY the JSON object."""

JD_PROMPT = """You are an expert HR data-extraction assistant.
Task: extract structured information from the JOB DESCRIPTION below.
Rules:
- required_skills = must-have skills; preferred_skills = nice-to-have / bonus skills.
- min_experience_years = smallest number of years asked for (0 if not mentioned).
- Use only information present in the text.

JOB DESCRIPTION:
<<<
{text}
>>>

{format_instructions}
Return ONLY the JSON object."""

REPAIR_PROMPT = """The JSON you produced is invalid.
Error: {error}

Your previous output:
{bad_output}

Return a corrected JSON object that strictly follows this schema:
{format_instructions}
Return ONLY the JSON object."""


def _llm_extract(text: str, model_cls: Type[BaseModel], prompt_template: str, llm) -> BaseModel:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=model_cls)
    fmt = parser.get_format_instructions()
    chain = ChatPromptTemplate.from_template(prompt_template) | llm
    raw = chain.invoke({"text": text[:12000], "format_instructions": fmt}).content

    for attempt in range(3):  # 1 parse + up to 2 repairs
        try:
            return model_cls.model_validate(extract_json(raw))
        except (ValueError, ValidationError, json.JSONDecodeError) as err:
            if attempt == 2:
                raise
            repair = ChatPromptTemplate.from_template(REPAIR_PROMPT) | llm
            raw = repair.invoke({"error": str(err)[:800], "bad_output": raw[:6000],
                                 "format_instructions": fmt}).content


# ----------------------------------------------------------------------------
# Offline rule-based extractors
# ----------------------------------------------------------------------------
HEADINGS = {
    "summary": ["summary", "profile", "about me", "objective", "about the role", "about"],
    "skills": ["skills", "technical skills", "tech stack"],
    "experience": ["experience", "work experience", "professional experience", "internships"],
    "education": ["education", "academics"],
    "projects": ["projects", "academic projects"],
    "certifications": ["certifications", "certificates", "courses"],
    "required": ["required skills", "requirements", "must have", "must-have skills"],
    "preferred": ["preferred skills", "nice to have", "good to have", "bonus"],
    "responsibilities": ["responsibilities", "what you will do", "key responsibilities"],
}


def _split_sections(text: str) -> dict:
    """Split text into sections using heading lines like 'SKILLS' or 'Skills:'."""
    sections, current = {"_top": []}, "_top"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, _, rest = stripped.partition(":")
        key = None
        for name, words in HEADINGS.items():
            if head.strip().lower() in words:
                key = name
                break
        if key:
            current = key
            sections.setdefault(current, [])
            if rest.strip():
                sections[current].append(rest.strip())
        else:
            sections.setdefault(current, []).append(stripped)
    return sections


def _bullets(lines):
    return [re.sub(r"^[-*•]\s*", "", l).strip() for l in lines if l.strip()]


def _years(s: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(years?|yrs?)", s, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*months?", s, re.I)
    return round(int(m.group(1)) / 12, 1) if m else 0.0


def rule_extract_resume(text: str) -> ResumeProfile:
    sec = _split_sections(text)
    top = sec.get("_top", [])
    name = re.sub(r"^name\s*:\s*", "", top[0], flags=re.I) if top else "Unknown"
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)

    experience = []
    for line in _bullets(sec.get("experience", [])):
        parts = [p.strip() for p in line.split("|")]
        experience.append(Experience(
            title=parts[0], company=parts[1] if len(parts) > 1 else "",
            years=_years(line), description=parts[-1] if len(parts) > 3 else ""))
    total = round(sum(e.years for e in experience), 1)

    return ResumeProfile(
        name=name.strip(),
        email=email.group(0) if email else None,
        total_experience_years=total,
        skills=find_skills_in_text(text),
        education=_bullets(sec.get("education", [])),
        experience=experience,
        projects=_bullets(sec.get("projects", [])),
        certifications=_bullets(sec.get("certifications", [])),
        summary=" ".join(sec.get("summary", []))[:400],
    )


def rule_extract_jd(text: str) -> JobProfile:
    sec = _split_sections(text)
    title, company, min_exp, education = "Unknown role", "", 0.0, []
    for line in sec.get("_top", []):
        k, _, v = line.partition(":")
        k = k.lower().strip()
        if k in ("job title", "title", "role", "position"):
            title = v.strip()
        elif k == "company":
            company = v.strip()
        elif k in ("experience", "min experience"):
            min_exp = _years(v)
    for line in sec.get("experience", []):   # "Experience: 2+ years" is caught as a heading
        min_exp = min_exp or _years(line)
    edu_lines = _bullets(sec.get("education", []))
    education = edu_lines

    req_text = " ".join(sec.get("required", []))
    pref_text = " ".join(sec.get("preferred", []))
    required = find_skills_in_text(req_text) if req_text else find_skills_in_text(text)
    preferred = [s for s in find_skills_in_text(pref_text) if s not in required]

    return JobProfile(
        title=title, company=company, min_experience_years=min_exp,
        required_skills=required, preferred_skills=preferred,
        education_requirements=education,
        responsibilities=_bullets(sec.get("responsibilities", [])),
        summary=" ".join(sec.get("summary", []))[:400],
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def extract_resume(text: str, use_llm: bool = True) -> Tuple[ResumeProfile, str]:
    """Returns (profile, mode) where mode is 'llm' or 'offline'."""
    llm = get_llm() if use_llm else None
    if llm is not None:
        try:
            p = _llm_extract(text, ResumeProfile, RESUME_PROMPT, llm)
            p.skills = normalize_skills(p.skills)
            return p, "llm"
        except Exception as e:
            print(f"[extract] LLM failed ({e}); using offline extractor")
    return rule_extract_resume(text), "offline"


def extract_jd(text: str, use_llm: bool = True) -> Tuple[JobProfile, str]:
    llm = get_llm() if use_llm else None
    if llm is not None:
        try:
            p = _llm_extract(text, JobProfile, JD_PROMPT, llm)
            p.required_skills = normalize_skills(p.required_skills)
            p.preferred_skills = [s for s in normalize_skills(p.preferred_skills)
                                  if s not in p.required_skills]
            return p, "llm"
        except Exception as e:
            print(f"[extract] LLM failed ({e}); using offline extractor")
    return rule_extract_jd(text), "offline"


def extract_cached(kind: str, key: str, text: str, use_llm: bool = True):
    """Extract once and cache to data/processed/<mode>/<kind>/<key>.json (saves API calls).
    LLM results and offline results are cached separately so they can be compared."""
    from .llm import llm_available
    model_cls = ResumeProfile if kind == "resumes" else JobProfile
    mode = "llm" if (use_llm and llm_available()) else "offline"
    path = CACHE_DIR / mode / kind / f"{key}.json"
    if path.exists():
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    fn = extract_resume if kind == "resumes" else extract_jd
    profile, got_mode = fn(text, use_llm=use_llm)
    if got_mode != mode:   # LLM failed this time -> don't pollute the LLM cache
        return profile
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def profiles_to_csv_rows(profiles):
    """Flatten resume profiles into CSV-friendly dicts."""
    rows = []
    for p in profiles:
        d = p.model_dump()
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = "; ".join(x if isinstance(x, str) else
                                 f"{x.get('title')} @ {x.get('company')} ({x.get('years')}y)" for x in v)
        rows.append(d)
    return rows
