"""
llm.py - Creates the LLM used for extraction and skill-gap analysis.

We use Groq (free tier) through LangChain. If no API key is set, or the call
fails, every module falls back to an OFFLINE rule-based mode, so the demo
never breaks.
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"   # Groq free-tier model (Nov 2026)


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip()) and os.getenv("OFFLINE_MODE", "0") != "1"


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model, or None when running offline."""
    if not llm_available():
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            temperature=temperature,
            max_retries=2,
            timeout=60,
        )
    except Exception as e:  # missing package / bad key format
        print(f"[llm] could not create LLM, using offline mode: {e}")
        return None


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ```json fences, extra text)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(text[start:end + 1])
