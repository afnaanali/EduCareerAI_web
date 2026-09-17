"""
core/rag_engine.py -- Lightweight Document RAG Engine for EduCareerAI
Provides PDF/DOCX/TXT extraction, text chunking, TF-IDF indexing,
and cosine-similarity retrieval -- zero heavy embedding model dependencies.

Designed for free Streamlit Community Cloud deployment.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MAX_FILE_SIZE_MB: int = 15
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 60
MAX_CHUNKS_PER_SESSION: int = 800
TOP_K_CHUNKS: int = 4
RELEVANCE_THRESHOLD: float = 0.08


def _extract_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        return pages
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def _extract_docx(file_bytes: bytes) -> List[Tuple[int, str]]:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        pages: List[Tuple[int, str]] = []
        current: List[str] = []
        current_words = 0
        page_num = 1
        for para in paragraphs:
            current.append(para)
            current_words += len(para.split())
            if current_words >= 500:
                pages.append((page_num, "\n".join(current)))
                page_num += 1
                current = []
                current_words = 0
        if current:
            pages.append((page_num, "\n".join(current)))
        return pages if pages else [(1, "\n".join(paragraphs))]
    except Exception as exc:
        raise RuntimeError(f"DOCX extraction failed: {exc}") from exc


def _extract_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            words = text.split()
            pages: List[Tuple[int, str]] = []
            chunk_words = 500
            for i in range(0, len(words), chunk_words):
                pages.append((len(pages) + 1, " ".join(words[i: i + chunk_words])))
            return pages if pages else [(1, text)]
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode text file with any supported encoding.")


def extract_text_from_file(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise RuntimeError(
            f"File '{filename}' is {len(file_bytes) // (1024 * 1024):.1f} MB -- "
            f"exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return _extract_pdf(file_bytes)
    elif ext == "docx":
        return _extract_docx(file_bytes)
    elif ext == "txt":
        return _extract_txt(file_bytes)
    else:
        raise RuntimeError(f"Unsupported file type '.{ext}'. Upload PDF, DOCX, or TXT.")


def _clean_text(text: str) -> str:
    """Cleans raw extracted text by removing noise, watermarks, URLs, and artifacts."""
    # Remove author/institute headers & watermark artifacts
    text = re.sub(r'\d*\s*S\d+\-SS Notes,?\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Prepared by\s+[A-Za-z\s\.,]+?(?=TYPES|SOFTWARE|HARDWARE|1\.|c\.|\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Asst\.?\s*Prof\.?[^,\n]*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'CSE Dept\.?[^,\n]*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'SNGIST[^,\n]*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'For More Study Materials\s*:?\s*https?://\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'keralanotes\.com', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_query(query: str) -> str:
    """Normalizes common typos and phrasing in student queries."""
    q = query.lower().strip()
    replacements = [
        (r'\bwhat\s+re\b', 'what are'),
        (r'\bwaht\b', 'what'),
        (r'\bwht\b', 'what'),
        (r'\bwat\b', 'what'),
        (r'\bwich\b', 'which'),
        (r'\bdif+ren+ce\b', 'difference'),
        (r'\btypes?\s+of\s+', 'types of '),
        (r'\btell\s+me\s+bout\b', 'tell me about'),
        (r'\bexplian\b', 'explain'),
        (r'\bdefination\b', 'definition'),
    ]
    for pattern, repl in replacements:
        q = re.sub(pattern, repl, q)
    return q


def chunk_text(pages: List[Tuple[int, str]], filename: str,
               chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    global_idx = 0
    for page_num, raw_text in pages:
        text = _clean_text(raw_text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_str = text[start:end].strip()
            if len(chunk_str) > 20:
                chunks.append({"text": chunk_str, "filename": filename,
                                "page": page_num, "chunk_idx": global_idx})
                global_idx += 1
            if end >= len(text):
                break
            start += chunk_size - overlap
    return chunks


def build_tfidf_index(chunks: List[Dict[str, Any]]) -> Tuple[Optional[TfidfVectorizer], Optional[Any]]:
    if not chunks:
        return None, None
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.95, min_df=1,
        ngram_range=(1, 2), stop_words="english", max_features=10000,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix
    except Exception:
        return None, None


def search_relevant_chunks(query: str, vectorizer: TfidfVectorizer, matrix: Any,
                            chunks: List[Dict[str, Any]], top_k: int = TOP_K_CHUNKS,
                            threshold: float = RELEVANCE_THRESHOLD) -> List[Dict[str, Any]]:
    if vectorizer is None or matrix is None or not chunks:
        return []
    try:
        norm_query = normalize_query(query)
        query_vec = vectorizer.transform([norm_query])
        scores = cosine_similarity(query_vec, matrix).flatten()
        candidate_indices = np.where(scores >= threshold)[0]
        if len(candidate_indices) == 0:
            return []
        sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        top_indices = sorted_indices[:top_k]
        results = []
        for idx in top_indices:
            chunk = dict(chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results
    except Exception:
        return []


def format_doc_context_for_prompt(relevant_chunks: List[Dict[str, Any]]) -> str:
    if not relevant_chunks:
        return ""
    parts = ["--- DOCUMENT CONTEXT (use when relevant to the user question) ---"]
    for i, chunk in enumerate(relevant_chunks, 1):
        clean_chunk = _clean_text(chunk['text'])
        parts.append(f"\n[Source {i}: {chunk['filename']} | Page {chunk['page']}]\n{clean_chunk}")
    parts.append("--- END DOCUMENT CONTEXT ---")
    return "\n".join(parts)


def build_source_attribution(relevant_chunks: List[Dict[str, Any]]) -> str:
    if not relevant_chunks:
        return ""
    seen: set = set()
    sources: List[str] = []
    for chunk in relevant_chunks:
        key = (chunk["filename"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"  - 📄 **{chunk['filename']}** — Page {chunk['page']}")
    if not sources:
        return ""
    return "\n\n---\n📚 **Sources**\n" + "\n".join(sources)


def synthesize_document_answer(user_query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Intelligently comprehends and synthesizes clean, structured educational answers
    from uploaded document chunks without dumping noisy raw OCR text.
    """
    if not relevant_chunks:
        return ""

    norm_q = normalize_query(user_query)
    combined_text = " ".join([_clean_text(c["text"]) for c in relevant_chunks])
    primary_filename = relevant_chunks[0]["filename"]

    # ── 1. Check for TYPES / CLASSIFICATION / CATEGORIES queries ───────────
    is_types_query = any(w in norm_q for w in ["type", "types", "kinds", "category", "categories", "classification", "classify", "divided in"])
    if is_types_query:
        # Match TYPES OF <X> or CLASSIFICATION OF <X>
        patterns = [
            r'TYPES OF\s+([A-Z\s]+?)(?=\s*\d|\s*[a-z]\.|\:|$)([\s\S]*?)(?=(?:TYPES OF|[A-Z\s]{4,}vs|SOFTWARE vs HARDWARE|\Z))',
            r'CLASSIFICATION OF\s+([A-Z\s]+?)(?=\s*\d|\s*[a-z]\.|\:|$)([\s\S]*?)(?=(?:TYPES OF|\Z))',
            r'CATEGORIES OF\s+([A-Z\s]+?)(?=\s*\d|\s*[a-z]\.|\:|$)([\s\S]*?)(?=(?:TYPES OF|\Z))',
        ]
        for pat in patterns:
            m = re.search(pat, combined_text, re.IGNORECASE)
            if m:
                heading = m.group(1).strip().title()
                body = m.group(2).strip()
                items = re.findall(
                    r'(?:(\d+)[\.\)]|\b([a-d])[\.\)])\s*([^:\-\n]+?)\s*[:\-]\s*([\s\S]*?)(?=(?:\d+[\.\)]|\b[a-d][\.\)]|\Z))',
                    body
                )
                if items:
                    icons = ["⚙️", "📱", "🛠️", "💾", "🌐", "🔒"]
                    resp = f"### 📚 Types of {heading} (From Your Uploaded Notes)\n\n"
                    resp += f"According to your uploaded document (**{primary_filename}**), **{heading}** is divided into the following primary categories:\n\n"
                    for idx, (n, l, name, desc) in enumerate(items):
                        icon = icons[idx % len(icons)]
                        clean_name = name.strip()
                        clean_desc = desc.strip()
                        # Extract examples if present
                        eg_match = re.search(r'(?:Eg|Example|Examples)\s*[:\-\s]+(.*)', clean_desc, re.IGNORECASE)
                        examples_str = ""
                        if eg_match:
                            examples_str = re.sub(r'^[:\-\s]+', '', eg_match.group(1)).strip()
                            clean_desc = clean_desc[:eg_match.start()].strip()

                        # Prevent trailing sentence bleed from next chunks
                        s_parts = [s.strip() for s in re.split(r'(?<=\.)\s+', clean_desc) if s.strip()]
                        valid_s = []
                        for s in s_parts:
                            if re.match(r'^[a-z]{1,3}\s', s):
                                break
                            valid_s.append(s)
                        if valid_s:
                            clean_desc = " ".join(valid_s)

                        resp += f"#### {idx + 1}. {icon} {clean_name}\n"
                        resp += f"- **Definition & Role:** {clean_desc}\n"
                        if examples_str:
                            resp += f"- **Examples:** `{examples_str}`\n"
                        resp += "\n"

                    resp += "---\n*💡 Extracted directly from your course material.*"
                    return resp

    # ── 2. Check for COMPARISON / DIFFERENCE / VS queries ───────────────────
    is_diff_query = any(w in norm_q for w in ["difference", "vs", "versus", "compare", "contrast", "between"])
    if is_diff_query:
        m = re.search(r'\b([A-Za-z]{2,}(?:\s+[A-Za-z]{2,})?)\s+vs\s+([A-Za-z]{2,}(?:\s+[A-Za-z]{2,})?)\b', combined_text, re.IGNORECASE)
        if m:
            raw_t1, raw_t2 = m.group(1).strip().title(), m.group(2).strip().title()
            t1 = raw_t1.split()[-1] if len(raw_t1.split()) > 1 and raw_t1.split()[-1].lower() in ["software", "hardware", "compiler", "interpreter", "memory", "storage", "system"] else raw_t1
            t2 = raw_t2.split()[0] if len(raw_t2.split()) > 1 and raw_t2.split()[0].lower() in ["software", "hardware", "compiler", "interpreter", "memory", "storage", "system"] else raw_t2
            # Find body after the match
            body = combined_text[m.end():]
            pts = re.findall(r'(\d+)[\.\)]\s*([\s\S]*?)(?=(?:\d+[\.\)]|[A-Z\s]{4,}:|\Z))', body)
            if pts:
                resp = f"### ⚖️ Comparison: {t1} vs {t2} (From Your Uploaded Notes)\n\n"
                resp += f"Based on **{primary_filename}**, here is the key comparison between **{t1}** and **{t2}**:\n\n"
                resp += f"| No. | Key Aspect & Comparison Points |\n"
                resp += f"| :--- | :--- |\n"
                for num, pt in pts[:8]:
                    clean_pt = pt.strip()
                    # Clean sentence bleed
                    s_parts = [s.strip() for s in re.split(r'(?<=\.)\s+', clean_pt) if s.strip()]
                    valid_s = [s for s in s_parts if not re.match(r'^[a-z]{1,3}\s', s)]
                    if valid_s:
                        clean_pt = " ".join(valid_s)
                    if clean_pt and len(clean_pt) > 5:
                        resp += f"| **{num}** | {clean_pt} |\n"
                resp += "\n---\n*💡 Formatted from your uploaded comparative notes.*"
                return resp

    # ── 3. Check for DEFINITION / WHAT IS queries ───────────────────────────
    # Look for matching term definition
    query_terms = [w for w in norm_q.replace("what is", "").replace("what are", "").replace("define", "").replace("explain", "").split() if len(w) > 2]
    for term in query_terms:
        pat = rf'\b({re.escape(term)}[a-z\s]*?)\s*[:\-]\s*([\s\S]*?)(?=(?:\d+[\.\)]|\b[a-d][\.\)]|[A-Z\s]{{4,}}:|\Z))'
        m = re.search(pat, combined_text, re.IGNORECASE)
        if m:
            def_title = m.group(1).strip().title()
            def_body = m.group(2).strip()
            if len(def_body) > 15:
                # Extract examples if present
                eg_match = re.search(r'(?:Eg|Example|Examples)\s*[:\-\s]+(.*)', def_body, re.IGNORECASE)
                examples_str = ""
                if eg_match:
                    examples_str = re.sub(r'^[:\-\s]+', '', eg_match.group(1)).strip()
                    def_body = def_body[:eg_match.start()].strip()

                resp = f"### 💡 {def_title} (From Your Uploaded Notes)\n\n"
                resp += f"Based on **{primary_filename}**:\n\n"
                resp += f"- **Definition:** {def_body}\n"
                if examples_str:
                    resp += f"- **Examples in Document:** `{examples_str}`\n"
                resp += "\n---\n*💡 Extracted directly from your study material.*"
                return resp

    # ── 4. General clean sentence extraction ────────────────────────────────
    sentences = re.split(r'(?<=[.!?])\s+', combined_text)
    matched_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 25 and any(term in s_clean.lower() for term in query_terms if len(term) > 3):
            matched_sentences.append(s_clean)

    if matched_sentences:
        resp = f"### 📖 Document Analysis: *\"{user_query}\"*\n\n"
        resp += f"Here are the core insights from **{primary_filename}**:\n\n"
        for s in matched_sentences[:4]:
            resp += f"- {s}\n"
        resp += "\n---\n*💡 Derived from your uploaded study document.*"
        return resp

    # Fallback to general clean excerpts if no specific rule matched
    excerpts = []
    for chunk in relevant_chunks[:2]:
        c_text = _clean_text(chunk["text"])
        if len(c_text) > 30:
            excerpts.append(f"> {c_text[:280]}...")

    return (
        f"### 📄 Insights from Your Uploaded Notes\n\n"
        + "\n\n".join(excerpts)
        + f"\n\n*(From `{primary_filename}`)*"
    )


def process_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], str]:
    try:
        pages = extract_text_from_file(file_bytes, filename)
        if not pages:
            return [], f"No readable text found in '{filename}'."
        chunks = chunk_text(pages, filename)
        if not chunks:
            return [], f"Could not split '{filename}' into searchable chunks."
        word_count = sum(len(c["text"].split()) for c in chunks)
        return chunks, f"Ready ({len(chunks)} chunks, ~{word_count} words)"
    except RuntimeError as exc:
        return [], str(exc)
    except Exception as exc:
        return [], f"Unexpected error processing '{filename}': {exc}"


def rebuild_index_from_chunks(all_chunks: List[Dict[str, Any]]) -> Tuple[Optional[TfidfVectorizer], Optional[Any]]:
    return build_tfidf_index(all_chunks)

