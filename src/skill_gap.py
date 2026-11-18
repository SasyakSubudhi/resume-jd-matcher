"""
skill_gap.py - Prompt chain that writes a skill-gap report per shortlisted candidate (Deliverable 3).

Chain (LangChain LCEL):
    Step 0 (code)  : compute matched / missing skills deterministically (no hallucination)
    Step 1 (LLM)   : ANALYSE  -> strengths, gap severity, experience gap, verdict (JSON)
    Step 2 (LLM)   : RECOMMEND -> concrete recommendation per gap + 30/60/90-day learning plan (JSON)
    Output         : validated SkillGapReport (Pydantic)

If the LLM is unavailable, a template-based report is produced instead.
"""
import json

from .llm import extract_json, get_llm
from .ranking import skill_match
from .schema import JobProfile, ResumeProfile, SkillGap, SkillGapReport

ANALYSE_PROMPT = """You are a senior technical recruiter.
Compare the CANDIDATE with the JOB. The skill comparison has already been computed for you
and is correct - do not contradict it.

JOB (JSON): {jd}
CANDIDATE (JSON): {resume}
MATCHED SKILLS: {matched}
MISSING REQUIRED SKILLS: {missing_required}
MISSING PREFERRED SKILLS: {missing_preferred}
SKILL MATCH SCORE (0-1): {skill_score}

Return ONLY a JSON object with keys:
"strengths": list of 3 short strings (why this candidate fits),
"gaps": list of objects {{"skill": str, "importance": "required"|"preferred", "severity": "high"|"medium"|"low"}}
        covering every missing skill (missing required = high or medium),
"experience_gap": one sentence comparing their experience with the requirement,
"verdict": "Strong fit" | "Moderate fit" | "Weak fit"."""

RECOMMEND_PROMPT = """You are a career coach for software engineers.
Given this skill-gap analysis for the role "{job_title}":
{analysis}

Return ONLY a JSON object with keys:
"recommendations": object mapping each gap skill to ONE concrete action
                   (a specific course, certification or mini-project, max 25 words),
"learning_plan": list of 3-5 ordered steps, each starting with "Days 1-30:", "Days 31-60:" or "Days 61-90:"."""


def _verdict(score: float) -> str:
    return "Strong fit" if score >= 0.75 else "Moderate fit" if score >= 0.5 else "Weak fit"


def _offline_report(resume: ResumeProfile, jd: JobProfile, sm: dict) -> SkillGapReport:
    gaps = [SkillGap(skill=s, importance="required", severity="high",
                     recommendation=f"Complete a hands-on course on {s} and build one small project using it.")
            for s in sm["missing_required"]]
    gaps += [SkillGap(skill=s, importance="preferred", severity="low",
                      recommendation=f"Learn the basics of {s} through official documentation/tutorials.")
             for s in sm["missing_preferred"]]
    exp_req, exp_have = jd.min_experience_years, resume.total_experience_years
    exp_txt = (f"Has {exp_have} years vs {exp_req} required - meets the requirement."
               if exp_have >= exp_req else f"Has {exp_have} years vs {exp_req} required - short by {round(exp_req - exp_have, 1)} years.")
    plan = []
    req = sm["missing_required"]
    if req:
        plan.append(f"Days 1-30: Learn {', '.join(req[:2])} and finish one guided project.")
    if len(req) > 2:
        plan.append(f"Days 31-60: Cover {', '.join(req[2:])} and add them to an end-to-end project.")
    if sm["missing_preferred"]:
        plan.append(f"Days 61-90: Pick up {', '.join(sm['missing_preferred'][:3])} and publish the project on GitHub.")
    if not plan:
        plan = ["Days 1-30: Candidate already covers all listed skills - focus on interview preparation and system design."]
    return SkillGapReport(
        candidate=resume.name, job_title=jd.title, fit_score=round(sm["score"] * 100, 1),
        verdict=_verdict(sm["score"]),
        strengths=[f"Has {len(sm['matched'])} of the job's listed skills: {', '.join(sm['matched'][:6])}",
                   f"{exp_have} years of relevant experience"] + ([resume.experience[0].title + " background"] if resume.experience else []),
        matched_skills=sm["matched"], gaps=gaps, experience_gap=exp_txt, learning_plan=plan)


def generate_skill_gap_report(resume: ResumeProfile, jd: JobProfile, use_llm: bool = True):
    """Returns (SkillGapReport, mode)."""
    sm = skill_match(resume, jd)
    llm = get_llm(temperature=0.2) if use_llm else None
    if llm is None:
        return _offline_report(resume, jd, sm), "offline"
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda

        analyse = ChatPromptTemplate.from_template(ANALYSE_PROMPT) | llm | StrOutputParser() | RunnableLambda(extract_json)
        recommend = ChatPromptTemplate.from_template(RECOMMEND_PROMPT) | llm | StrOutputParser() | RunnableLambda(extract_json)

        analysis = analyse.invoke({
            "jd": jd.model_dump_json(), "resume": resume.model_dump_json(),
            "matched": sm["matched"], "missing_required": sm["missing_required"],
            "missing_preferred": sm["missing_preferred"], "skill_score": round(sm["score"], 2)})
        rec = recommend.invoke({"job_title": jd.title, "analysis": json.dumps(analysis)})

        recs = rec.get("recommendations", {})
        gaps = [SkillGap(skill=g["skill"], importance=g.get("importance", "required"),
                         severity=g.get("severity", "medium"),
                         recommendation=recs.get(g["skill"], "Take a course and build a small project."))
                for g in analysis.get("gaps", [])]
        report = SkillGapReport(
            candidate=resume.name, job_title=jd.title, fit_score=round(sm["score"] * 100, 1),
            verdict=analysis.get("verdict", _verdict(sm["score"])),
            strengths=analysis.get("strengths", []), matched_skills=sm["matched"], gaps=gaps,
            experience_gap=analysis.get("experience_gap", ""),
            learning_plan=rec.get("learning_plan", []))
        return report, "llm"
    except Exception as e:
        print(f"[skill_gap] LLM chain failed ({e}); using offline report")
        return _offline_report(resume, jd, sm), "offline"


def report_to_markdown(r: SkillGapReport) -> str:
    lines = [f"## Skill-gap report: {r.candidate} -> {r.job_title}",
             f"**Fit score:** {r.fit_score}/100  |  **Verdict:** {r.verdict}", "",
             "**Strengths**"] + [f"- {s}" for s in r.strengths]
    lines += ["", f"**Matched skills:** {', '.join(r.matched_skills) or 'none'}", "", "**Gaps**"]
    lines += [f"- **{g.skill}** ({g.importance}, {g.severity}): {g.recommendation}" for g in r.gaps] or ["- None"]
    lines += ["", f"**Experience:** {r.experience_gap}", "", "**Learning plan**"]
    lines += [f"- {s}" for s in r.learning_plan]
    return "\n".join(lines)
