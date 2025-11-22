import hashlib
import re


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


