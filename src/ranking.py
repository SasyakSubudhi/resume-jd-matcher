"""
ranking.py - Ranks candidates for a job description (Deliverable 2).

Methods compared:
    glove  : cosine similarity of averaged GloVe vectors (static embeddings)
    sbert  : cosine similarity of Sentence-Transformer vectors from ChromaDB (contextual)
    hybrid : 0.6 * sbert similarity + 0.4 * skill-match score   (our final system)

Skill-match score (0..1) = 0.75 * required-skill coverage
                         + 0.25 * preferred-skill coverage
                         - small penalty if experience < minimum required.
"""
import hashlib
from typing import Dict, List

from .embeddings import EMBEDDERS
from .schema import JobProfile, ResumeProfile
from .skills import skill_set
from .vectorstore import build_collection, query_collection

HYBRID_W_SEM = 0.6
HYBRID_W_SKILL = 0.4


def resume_to_text(p: ResumeProfile) -> str:
    exp = ". ".join(f"{e.title} at {e.company}: {e.description}" for e in p.experience)
    return (f"Skills: {', '.join(p.skills)}. Experience: {exp}. "
            f"Projects: {'. '.join(p.projects)}. {p.summary}")


def jd_to_text(j: JobProfile) -> str:
    return (f"{j.title}. Required skills: {', '.join(j.required_skills)}. "
            f"Preferred skills: {', '.join(j.preferred_skills)}. "
            f"Responsibilities: {'. '.join(j.responsibilities)}. {j.summary}")


def skill_match(resume: ResumeProfile, jd: JobProfile) -> Dict:
    have = skill_set(resume.skills)
    req = skill_set(jd.required_skills)
    pref = skill_set(jd.preferred_skills)
    req_cov = len(have & req) / len(req) if req else 1.0
    pref_cov = len(have & pref) / len(pref) if pref else 0.0
    score = 0.75 * req_cov + 0.25 * pref_cov
    if jd.min_experience_years and resume.total_experience_years < jd.min_experience_years:
        shortfall = (jd.min_experience_years - resume.total_experience_years) / jd.min_experience_years
        score -= 0.15 * shortfall
    canon = {s.lower(): s for s in list(jd.required_skills) + list(jd.preferred_skills)}
    return {
        "score": max(0.0, min(1.0, score)),
        "required_coverage": req_cov,
        "preferred_coverage": pref_cov,
        "matched": [canon[s] for s in canon if s in have],
        "missing_required": [s for s in jd.required_skills if s.lower() not in have],
        "missing_preferred": [s for s in jd.preferred_skills if s.lower() not in have],
    }


class CandidateRanker:
    """Indexes all resumes into ChromaDB (one collection per embedding model)."""

    def __init__(self, resumes: Dict[str, ResumeProfile]):
        self.resumes = resumes
        self.ids = list(resumes.keys())
        self.docs = [resume_to_text(resumes[i]) for i in self.ids]
        self.meta = [{"name": resumes[i].name,
                      "experience_years": float(resumes[i].total_experience_years),
                      "skills": ", ".join(resumes[i].skills)} for i in self.ids]
        # unique collection suffix per candidate pool, so different pools never clash
        self.tag = hashlib.md5("|".join(self.ids + self.docs).encode()).hexdigest()[:10]
        self._built = set()

    def _ensure_index(self, method: str):
        if method not in self._built:
            emb = EMBEDDERS[method](self.docs)
            build_collection(f"resumes_{method}_{self.tag}", self.ids, emb, self.docs, self.meta)
            self._built.add(method)

    def rank(self, jd: JobProfile, method: str = "hybrid") -> List[Dict]:
        """Return ALL candidates sorted best-first with their scores."""
        sem_method = "sbert" if method == "hybrid" else method
        self._ensure_index(sem_method)
        q = EMBEDDERS[sem_method]([jd_to_text(jd)])[0]
        hits = query_collection(f"resumes_{sem_method}_{self.tag}", q, k=len(self.ids))

        rows = []
        for h in hits:
            sm = skill_match(self.resumes[h["id"]], jd)
            sem = float(h["similarity"])
            final = HYBRID_W_SEM * sem + HYBRID_W_SKILL * sm["score"] if method == "hybrid" else sem
            rows.append({
                "id": h["id"], "name": h["metadata"]["name"],
                "experience_years": h["metadata"]["experience_years"],
                "semantic_similarity": round(sem, 4),
                "skill_match": round(sm["score"], 4),
                "score": round(final, 4),
                "matched_skills": sm["matched"],
                "missing_required": sm["missing_required"],
                "missing_preferred": sm["missing_preferred"],
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows
