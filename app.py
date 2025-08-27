import os
import io
import re
import streamlit as st
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

import srt
from datetime import timedelta

# Replace with your chosen providers/wrappers
from openai import OpenAI

# FAISS + embeddings (LangChain wrappers)
# pip install faiss-cpu langchain-community langchain-openai tiktoken
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ------------- Config -------------
SUPPORTED_LANGUAGES = {
    "Spanish (es)": "es",
    "German (de)": "de",
    "Japanese (ja)": "ja",
    "French (fr)": "fr",
    "Hindi (hi)": "hi",
}

DEFAULT_STYLE = "Neutral, clear, and faithful to the source. Preserve meaning precisely."

# ------------- Utilities -------------

def detect_filetype(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".srt"):
        return "srt"
    elif lower.endswith(".vtt"):
        return "vtt"  # optional: implement vtt
    elif lower.endswith(".txt"):
        return "txt"
    else:
        return "txt"  # fallback treat as text

def read_uploaded_file(uploaded_file) -> str:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return content

def parse_srt(content: str) -> List[srt.Subtitle]:
    return list(srt.parse(content))

def srt_to_text_blocks(subs: List[srt.Subtitle]) -> List[Tuple[str, Tuple[int,int,int]]]:
    # returns list of (text, (index, start_ms, end_ms))
    blocks = []
    for idx, sub in enumerate(subs, start=1):
        text = sub.content
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms = int(sub.end.total_seconds() * 1000)
        blocks.append((text, (idx, start_ms, end_ms)))
    return blocks

def text_to_blocks(text: str, max_chars=800) -> List[str]:
    # naive splitter by paragraphs, then chunk
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = f"{cur} {s}".strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            if cur:
                chunks.append(cur)
    return chunks

def ms_to_srt_time(ms: int) -> timedelta:
    return timedelta(milliseconds=ms)

def build_system_prompt(style: str) -> str:
    return f"""
You are a professional translator. Your goals:
- Translate the source text accurately into the target language.
- Preserve all formatting, tags, placeholders, and numbers.
- Do not add or remove information.
- If input is from subtitles, keep line breaks and avoid length inflation.
- Style: {style}
"""

def build_user_prompt(source_text: str, target_lang: str, rag_snippets: Optional[List[str]]=None, glossary: Optional[Dict[str, Dict[str,str]]]=None) -> str:
    gloss_text = ""
    if glossary:
        gloss_pairs = []
        for term, row in glossary.items():
            tgt = row.get(target_lang, "")
            if tgt:
                gloss_pairs.append(f"{term} -> {tgt}")
        if gloss_pairs:
            gloss_text = "Glossary mappings (use preferred terms):\n" + "\n".join(gloss_pairs)

    rag_text = ""
    if rag_snippets:
        rag_text = "Domain context (use to resolve terminology):\n" + "\n---\n".join(rag_snippets)

    return f"""
Target language: {target_lang}
Translate the following source text. Use provided glossary and context if present.
Source:
{source_text}

{gloss_text}

{rag_text}
"""

def load_glossary(file) -> Dict[str, Dict[str,str]]:
    # Expect CSV format: term,es,de,ja,fr
    import csv
    glossary = {}
    content = file.getvalue().decode("utf-8", errors="ignore").splitlines()
    reader = csv.DictReader(content)
    for row in reader:
        term = row.get("term") or row.get("Term") or row.get("source") or row.get("Source")
        if not term:
            continue
        glossary[term.strip()] = {
            "es": row.get("es","").strip(),
            "de": row.get("de","").strip(),
            "ja": row.get("ja","").strip(),
            "fr": row.get("fr","").strip(),
            "hi": row.get("hi","").strip(),
        }
    return glossary

# ------------- RAG with FAISS -------------

@st.cache_resource(show_spinner=False)
def build_rag_index_faiss(docs: list[str]):
    # Uses OpenAI embeddings; ensure your OPENAI_API_KEY is set on the server
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    metadatas = [{"id": i} for i in range(len(docs))]
    vs = FAISS.from_texts(docs, embedding=embeddings, metadatas=metadatas)
    return vs

def retrieve_context_faiss(vs, query: str, k: int = 3) -> list[str]:
    if vs is None:
        return []
    res = vs.similarity_search(query, k=k)
    return [r.page_content for r in res]

# ------------- LLM Client -------------

def make_client():
    # Option A (recommended): ensure OPENAI_API_KEY is set in your server environment
    # Option B (not recommended): hardcode your key here
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # return OpenAI()
    # Read the API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    # # Pass the key explicitly to OpenAI client
    return OpenAI(api_key=api_key)
    
# def translate_chunks(client, chunks: List[str], target_lang_code: str, style: str,
#                      rag_vs=None, glossary=None, model="gpt-4o-mini") -> List[str]:
#     outputs = []
#     system = build_system_prompt(style)
#     for ch in chunks:
#         context = retrieve_context_faiss(rag_vs, ch, k=3) if rag_vs is not None else []
#         user = build_user_prompt(ch, target_lang_code, rag_snippets=context, glossary=glossary)
#         resp = client.chat.completions.create(
#             model=model,
#             temperature=0.2,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user}
#             ]
#         )
#         outputs.append(resp.choices[0].message.content.strip())
#     return outputs
def translate_chunks(client, chunks: List[str], target_lang_code: str, style: str,
                     rag_vs=None, glossary=None, model="gpt-4o-mini") -> List[str]:
    outputs = []
    system = build_system_prompt(style)

    for ch in chunks:
        context = retrieve_context_faiss(rag_vs, ch, k=3) if rag_vs is not None else []
        user = build_user_prompt(ch, target_lang_code, rag_snippets=context, glossary=glossary)

        # Build kwargs dynamically
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }

        # Only set temperature if not GPT-5 family
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0.2

        resp = client.chat.completions.create(**kwargs)
        outputs.append(resp.choices[0].message.content.strip())

    return outputs


# ------------- Streamlit UI -------------

st.set_page_config(page_title="Multilingual Transcript Translator", page_icon="🌐", layout="wide")

st.title("Multilingual Transcript Translator (RAG-enabled, FAISS)")

with st.sidebar:
    st.header("Settings")
    # Removed API key input (uses server-side key)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini" ,
                                   "gpt-5",
                                    "gpt-5-mini",
                                    "gpt-5-nano",])
    style = st.text_area("Translation style/tone", value=DEFAULT_STYLE)
    rag_enabled = st.checkbox("Enable RAG (domain-aware context)", value=False)
    rag_docs = st.text_area("Domain documents (one block per line)", help="Paste domain-specific notes, terminology, or style guidance.")
    glossary_file = st.file_uploader("Glossary CSV (term,es,de,ja,fr,hi)", type=["csv"])

st.subheader("1) Upload transcript")
uploaded = st.file_uploader("Upload transcript file (.txt or .srt)", type=["txt", "srt"])

st.subheader("2) Choose target languages")
lang_labels = st.multiselect("Languages", list(SUPPORTED_LANGUAGES.keys()), default=["Spanish (es)", "German (de)", "Japanese (ja)", "French (fr)","Hindi (hi)"])

do_translate = st.button("Translate")

if do_translate:
    if not uploaded:
        st.error("Please upload a transcript file.")
        st.stop()

    # Parse file
    filetype = detect_filetype(uploaded.name)
    raw = read_uploaded_file(uploaded)

    srt_mode = (filetype == "srt")
    if srt_mode:
        try:
            subs = parse_srt(raw)
            blocks = srt_to_text_blocks(subs)  # list of (text, meta)
            source_blocks = [b[0] for b in blocks]
        except Exception as e:
            st.error(f"Failed to parse SRT: {e}")
            st.stop()
    else:
        source_blocks = text_to_blocks(raw)

    # Build FAISS RAG index if enabled and docs provided
    rag_vs = None
    if rag_enabled:
        docs = [d.strip() for d in rag_docs.splitlines() if d.strip()]
        if docs:
            try:
                rag_vs = build_rag_index_faiss(docs)
            except Exception as e:
                st.warning(f"Could not build RAG index: {e}")

    # Load glossary
    glossary = None
    if glossary_file:
        try:
            glossary = load_glossary(glossary_file)
        except Exception as e:
            st.warning(f"Could not parse glossary: {e}")

    client = make_client()

    # Translate per language
    for label in lang_labels:
        lang_code = SUPPORTED_LANGUAGES[label]
        st.write(f"Translating to {label}...")
        out_blocks = translate_chunks(client, source_blocks, lang_code, style, rag_vs, glossary, model=model)

        if srt_mode:
            # Reconstruct SRT with original timings
            new_subs = []
            for (content, (idx, start_ms, end_ms)), translated in zip(blocks, out_blocks):
                sub = srt.Subtitle(
                    index=idx,
                    start=ms_to_srt_time(start_ms),
                    end=ms_to_srt_time(end_ms),
                    content=translated
                )
                new_subs.append(sub)
            srt_output = srt.compose(new_subs)
            st.download_button(
                label=f"Download {label} SRT",
                data=srt_output.encode("utf-8"),
                file_name=f"translation_{SUPPORTED_LANGUAGES[label]}.srt",
                mime="application/x-subrip"
            )
            st.text_area(f"Preview ({label})", value=srt_output, height=300)
        else:
            full_text = "\n\n".join(out_blocks)
            st.download_button(
                label=f"Download {label} text",
                data=full_text.encode("utf-8"),
                file_name=f"translation_{SUPPORTED_LANGUAGES[label]}.txt",
                mime="text/plain"
            )
            st.text_area(f"Preview ({label})", value=full_text, height=300)