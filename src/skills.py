"""
skills.py - Skill taxonomy + normalisation.

Different resumes write the same skill differently ("JS", "Javascript",
"javascript"). We map every alias to ONE canonical name so that skill
matching between a resume and a job description is fair.
"""
import re
from typing import Iterable, List, Set

# canonical name -> list of aliases (all lowercase)
SKILL_ALIASES = {
    "Python": ["python", "python3"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C": ["c language", "c programming"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript", "ts"],
    "Go": ["golang", "go lang"],
    "SQL": ["sql"],
    "R": ["r programming", "r language"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Tailwind CSS": ["tailwind", "tailwind css", "tailwindcss"],
    "React": ["react", "react.js", "reactjs"],
    "Next.js": ["next.js", "nextjs"],
    "Redux": ["redux"],
    "Angular": ["angular", "angularjs"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Node.js": ["node", "node.js", "nodejs"],
    "Express": ["express", "express.js", "expressjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring Boot": ["spring boot", "springboot", "spring"],
    "REST APIs": ["rest", "rest api", "rest apis", "restful", "restful apis"],
    "GraphQL": ["graphql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv", "opencv"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow", "tf", "keras"],
    "PyTorch": ["pytorch", "torch"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "LangChain": ["langchain"],
    "LLMs": ["llm", "llms", "large language models", "gpt", "prompt engineering"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Excel": ["excel", "ms excel"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow", "apache airflow"],
    "Docker": ["docker", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "ci cd", "cicd", "github actions", "gitlab ci"],
    "Linux": ["linux", "unix", "bash", "shell scripting"],
    "Git": ["git", "github", "gitlab"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "Microservices": ["microservices", "microservice"],
    "System Design": ["system design"],
    "Unit Testing": ["unit testing", "pytest", "junit", "jest"],
    "Agile": ["agile", "scrum"],
    "Figma": ["figma"],
    "Data Structures & Algorithms": ["dsa", "data structures", "algorithms"],
    "MLOps": ["mlops", "mlflow"],
    "Kafka": ["kafka", "apache kafka"],
}

# reverse lookup: alias -> canonical
_ALIAS_TO_CANON = {}
for canon, aliases in SKILL_ALIASES.items():
    _ALIAS_TO_CANON[canon.lower()] = canon
    for a in aliases:
        _ALIAS_TO_CANON[a] = canon


def normalize_skill(skill: str) -> str:
    """Map any spelling of a skill to its canonical name (unknown skills are title-cased)."""
    s = skill.strip().lower().rstrip(".")
    return _ALIAS_TO_CANON.get(s, skill.strip())


def normalize_skills(skills: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for sk in skills:
        n = normalize_skill(sk)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def find_skills_in_text(text: str) -> List[str]:
    """Rule-based skill finder: scan raw text for any known alias (used in offline mode)."""
    low = " " + text.lower() + " "
    found = []
    for alias, canon in _ALIAS_TO_CANON.items():
        # word-boundary match; escape special chars like c++ / ci/cd
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if len(alias) <= 2 and alias not in ("go",):
            # very short aliases (ml, js, cv, tf) only count if written as a whole token
            pattern = r"(?<![a-z0-9+#./])" + re.escape(alias) + r"(?![a-z0-9+#])"
        if re.search(pattern, low):
            found.append(canon)
    return normalize_skills(found)


def skill_set(skills: Iterable[str]) -> Set[str]:
    return {s.lower() for s in normalize_skills(skills)}
