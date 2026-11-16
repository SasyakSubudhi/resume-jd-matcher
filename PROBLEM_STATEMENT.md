# Problem Statement 4: Resume and Job Description Matching System

**Course:** Jackfruit Problem, Mini-Project (PES University)
**Team (SRN):** PES1UG24AM420 · PES1UG24AM256 · PES1UG24AM266 · PES1UG24AM269

## Problem statement
Develop a system that extracts structured information from resumes and job descriptions,
ranks candidates by suitability using embedding search, and generates a skill-gap report
for each candidate.

## Deliverables and how we cover them

| # | Deliverable | Where it is in the repository |
|---|-------------|-------------------------------|
| 1 | Structured-output extraction module that converts resumes and job descriptions into a fixed JSON schema | `src/schema.py` (Pydantic schema), `src/extract.py` (LangChain prompt + `PydanticOutputParser` + automatic JSON repair + offline fallback) |
| 2 | Ranking module that compares static embeddings (GloVe) with contextual sentence embeddings stored in a vector database, evaluated with NDCG@5 against human rankings | `src/embeddings.py` (GloVe vs Sentence-Transformers), `src/vectorstore.py` (ChromaDB), `src/ranking.py`, `src/evaluate.py`, `data/human_rankings.json`, `scripts/run_evaluation.py` |
| 3 | Prompt chain that generates a skill-gap analysis and recommendations for each shortlisted candidate | `src/skill_gap.py` (LangChain LCEL chain: analyse → recommend) |
| — | Working application for the live demo | `app.py` (Streamlit) |

## Dataset
* 20 sample resumes (`data/resumes/`) covering data science, ML, backend, frontend,
  DevOps, QA and one non-software profile.
* 4 job descriptions (`data/jds/`): Data Scientist, Backend Developer (Python),
  Frontend Developer (React), DevOps Engineer.
* Human relevance grades from 0 to 3 for every (job, resume) pair (`data/human_rankings.json`).
* You can also upload real resumes (PDF, DOCX or TXT) in the app.
