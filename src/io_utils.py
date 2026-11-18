"""
io_utils.py - Read resumes / job descriptions from .txt, .pdf or .docx files.
"""
import io
from pathlib import Path


def read_text_from_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")


def read_text_file(path) -> str:
    path = Path(path)
    return read_text_from_bytes(path.read_bytes(), path.name)


def load_folder(folder) -> dict:
    """Return {file_stem: text} for every .txt/.pdf/.docx in a folder (sorted)."""
    folder = Path(folder)
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".txt", ".pdf", ".docx"):
            out[p.stem] = read_text_file(p)
    return out
