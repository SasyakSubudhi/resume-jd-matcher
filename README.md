# 🎯 Resume & Job Description Matching System

Extracts structured information from resumes and job descriptions, ranks candidates
with embedding search (static **GloVe** vs contextual **Sentence-Transformers** in
**ChromaDB**), evaluates the rankings with **NDCG@5** against human rankings, and writes
a **skill-gap report** for every shortlisted candidate using an LLM prompt chain.

**Team (SRN):** PES1UG24AM420 · PES1UG24AM256 · PES1UG24AM266 · PES1UG24AM269
**Problem statement:** see [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) · **Why & What:** [`docs/One_Page_Why_What.pdf`](docs/One_Page_Why_What.pdf)

---

## Architecture

```
 Resume / JD (PDF, DOCX, TXT)
          │
          ▼
 ┌──────────────────────────┐   LLM (Groq, via LangChain) + PydanticOutputParser
 │ 1. EXTRACTION            │── invalid JSON? → repair prompt (max 2 tries)
 │    src/extract.py        │── no API / still failing → offline rule-based extractor
 └──────────┬───────────────┘
            ▼   fixed JSON schema (src/schema.py)
 ┌──────────────────────────┐   GloVe (static, averaged word vectors)
 │ 2. RANKING               │   SBERT all-MiniLM-L6-v2 (contextual) → ChromaDB (cosine)
 │    src/ranking.py        │   Hybrid = 0.6 × SBERT similarity + 0.4 × skill match
 └──────────┬───────────────┘
            ▼   top-k shortlist                     ┌─────────────────────────────┐
 ┌──────────────────────────┐                       │ EVALUATION  src/evaluate.py │
 │ 3. SKILL-GAP CHAIN       │                       │ NDCG@5 vs human grades 0-3  │
 │    src/skill_gap.py      │                       └─────────────────────────────┘
 │ compute gaps → ANALYSE → RECOMMEND → SkillGapReport
 └──────────────────────────┘
            ▼
      Streamlit app (app.py)
```

## Results (NDCG@5, 4 jobs × 20 candidates)

| Method | Mean NDCG@5 | Mean P@5 |
|--------|------------:|---------:|
| Static: GloVe (glove-wiki-gigaword-50) | 0.807 | 0.40 |
| Contextual: SBERT + ChromaDB | 0.959 | 0.60 |
| **Hybrid: SBERT + skill match** | **0.961** | **0.60** |

Contextual embeddings beat static ones because GloVe gives each word one fixed vector,
averages away word order and has no vectors for many tech terms (for example "fastapi",
"pytorch" or "nextjs"). SBERT encodes the whole sentence in context. P@5 counts grade ≥ 2
candidates in the top 5. Each job has only 3–4 such candidates, so the maximum possible
P@5 is 0.6–0.8.
Regenerate with `python scripts/run_evaluation.py` → `results/`.

## Setup (Windows – Command Prompt)

```bat
cd resume-jd-matcher
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```
Paste your free Groq API key from https://console.groq.com/keys between the quotes and save.
The key is optional: without one, the app runs fully offline in rule-based mode.

### 🔒 How the API key is kept secret
* The key is **never written in the source code**. `src/llm.py` reads it at runtime from
  **Streamlit secrets** (`.streamlit/secrets.toml`), falling back to an environment variable / `.env`.
* `secrets.toml` and `.env` are in **`.gitignore`**, so the key can't be committed to GitHub.
  Only the empty templates (`secrets.toml.example`, `.env.example`) are in the repo.
* Inside the program the key is wrapped in **`pydantic.SecretStr`**, so it prints as
  `**********` if it ever ends up in a log or error message.
* Error messages print only the exception type, never request details that could contain the key.
* If a key is ever leaked, revoke it at console.groq.com/keys and create a new one.

```bat
streamlit run app.py
```
The first run downloads GloVe (~66 MB) and the SBERT model (~90 MB) once.
**Run the app once before the demo so the models are cached.**

Linux/macOS: use `source venv/bin/activate` and `cp .env.example .env`.

## Using the app
1. **Extract**: pick a sample or upload a resume/JD to see the JSON output. You can download it as JSON or CSV and add uploaded resumes to the candidate pool.
2. **Rank candidates**: pick a job and a method (GloVe / SBERT / Hybrid) to see the ranked table, scores and missing skills. One expander compares all three methods side by side.
3. **Skill-gap report**: pick shortlisted candidates to get strengths, gaps with severity, recommendations and a 30/60/90-day plan.
4. **Evaluation**: shows NDCG@5 per method and per job, plus the human ground-truth grades.

## Project structure
```
app.py                  Streamlit UI
src/schema.py           Pydantic JSON schema (ResumeProfile, JobProfile, SkillGapReport)
src/llm.py              Groq LLM factory + JSON extraction helper
src/extract.py          Structured extraction (LLM + repair + offline fallback)
src/skills.py           Skill taxonomy & alias normalisation
src/io_utils.py         PDF / DOCX / TXT reading
src/embeddings.py       GloVe and Sentence-Transformer embeddings
src/vectorstore.py      ChromaDB collections
src/ranking.py          GloVe / SBERT / Hybrid ranking + skill match
src/skill_gap.py        Skill-gap prompt chain
src/evaluate.py         NDCG@k, P@k
scripts/run_evaluation.py   Full experiment → results/
data/resumes, data/jds  Sample dataset
data/human_rankings.json    Human relevance grades (ground truth)
```

## Tech stack
Python · LangChain · Groq (`openai/gpt-oss-20b`) · Pydantic · Sentence-Transformers ·
GloVe · ChromaDB · Streamlit · pandas · matplotlib
