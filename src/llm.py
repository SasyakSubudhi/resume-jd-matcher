"""
llm.py - Creates the LLM used for extraction and skill-gap analysis.

We use Groq (free tier) through LangChain. If no API key is set, or the call
fails, every module falls back to an OFFLINE rule-based mode, so the demo
never breaks.

SECRETS HANDLING
----------------
The Groq API key is NEVER written in the code. It is read, in this order, from:
  1. Streamlit secrets  -> .streamlit/secrets.toml   (recommended)
  2. Environment / .env -> GROQ_API_KEY
Both files are listed in .gitignore, so the key can never be pushed to GitHub.
Inside the program the key is wrapped in pydantic.SecretStr, so it shows as
'**********' if it is ever printed or logged.
"""
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"   # Groq free-tier model (Nov 2026)


def _get_setting(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then from environment variables / .env."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:          # no secrets.toml, or not running inside Streamlit
        pass
    return os.getenv(name, default).strip()


def get_api_key() -> Optional[SecretStr]:
    key = _get_setting("GROQ_API_KEY")
    return SecretStr(key) if key else None


def llm_available() -> bool:
    return get_api_key() is not None and _get_setting("OFFLINE_MODE", "0") != "1"


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model, or None when running offline."""
    if not llm_available():
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=_get_setting("GROQ_MODEL", DEFAULT_MODEL),
            api_key=get_api_key(),        # passed as SecretStr, never as plain text
            temperature=temperature,
            max_retries=2,
            timeout=60,
        )
    except Exception as e:  # missing package / bad key format
        print(f"[llm] could not create LLM, using offline mode: {type(e).__name__}")
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
