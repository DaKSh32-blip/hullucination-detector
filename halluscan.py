#!/usr/bin/env python3
"""
HalluScan v2: claim-level hallucination detector for LLM output. Single-file version.

What it does
  1. Splits text into atomic, self-contained claims (spaCy dependency parse).
  2. Finds evidence on Wikipedia, the web, or a document you supply.
  3. Ranks evidence sentences by meaning (MiniLM embeddings).
  4. Judges each claim with an NLI model (DeBERTa-v3, trained on MNLI/FEVER/ANLI).
  5. Returns SUPPORTED / REFUTED / NOT_ENOUGH_INFO / UNVERIFIABLE per claim, with evidence links.

Setup
  pip install fastapi "uvicorn[standard]" python-multipart pydantic requests spacy \
              transformers torch sentence-transformers sentencepiece protobuf pypdf python-docx
  python -m spacy download en_core_web_sm

Usage
  python halluscan.py "Einstein was born in Paris in 1879."        # check text
  python halluscan.py --file answer.pdf                             # check a PDF/DOCX/TXT
  python halluscan.py "..." --context source.txt                    # RAG faithfulness mode
  python halluscan.py "..." --json                                  # raw JSON output
  python halluscan.py --serve --port 8000                           # API + demo page
        -> http://localhost:8000      demo page
        -> http://localhost:8000/docs interactive API docs

In Colab / Jupyter
  from halluscan import Detector
  det = Detector(); print(det.verify("Einstein was born in Paris.").summary)
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Set, Tuple

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

__version__ = "2.0.0"


# ============================================================================
# CONFIG
# ============================================================================
# Runtime settings. Every value can be overridden with an environment variable.


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HALLUSCAN_{name}", default)


@dataclass(frozen=True)
class Settings:
    # Models
    nli_model: str = field(default_factory=lambda: _env(
        "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"))
    embed_model: str = field(default_factory=lambda: _env(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    spacy_model: str = field(default_factory=lambda: _env("SPACY_MODEL", "en_core_web_sm"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cpu | cuda

    # Retrieval
    wiki_pages_per_claim: int = field(default_factory=lambda: int(_env("WIKI_PAGES", "3")))
    evidence_top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "8")))
    max_workers: int = field(default_factory=lambda: int(_env("MAX_WORKERS", "8")))

    # Decision thresholds (tuned so a claim needs clear evidence either way)
    entail_threshold: float = field(default_factory=lambda: float(_env("ENTAIL_T", "0.60")))
    contra_threshold: float = field(default_factory=lambda: float(_env("CONTRA_T", "0.60")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.25")))

    # API limits
    max_chars: int = field(default_factory=lambda: int(_env("MAX_CHARS", "20000")))
    max_claims: int = field(default_factory=lambda: int(_env("MAX_CLAIMS", "60")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "10")))
    preload_models: bool = field(default_factory=lambda: _env("PRELOAD", "1") == "1")

    user_agent: str = "HalluScan/2.0 (educational fact-checking project)"


settings = Settings()


# ============================================================================
# SCHEMAS
# ============================================================================
# Request and response shapes shared by the pipeline and the API.


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"          # evidence entails the claim
    REFUTED = "REFUTED"              # evidence contradicts the claim -> likely hallucination
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"  # evidence found, but it neither confirms nor denies
    UNVERIFIABLE = "UNVERIFIABLE"    # no relevant evidence found at all


Source = Literal["wikipedia", "context", "web"]


class Claim(BaseModel):
    id: int
    text: str = Field(description="Self-contained atomic claim (pronouns resolved).")
    sentence: str = Field(description="Original sentence the claim came from.")
    start: int = Field(description="Char offset of the source sentence in the input.")
    end: int
    entities: List[str] = []


class Evidence(BaseModel):
    text: str
    source: Source
    title: Optional[str] = None
    url: Optional[str] = None
    relevance: float = Field(description="Semantic similarity between claim and evidence, 0..1")
    entailment: float
    contradiction: float
    neutral: float


class ClaimResult(BaseModel):
    claim: Claim
    verdict: Verdict
    confidence: float
    best_evidence: Optional[Evidence] = None
    evidence: List[Evidence] = []
    flags: List[str] = []


class Summary(BaseModel):
    total_claims: int
    supported: int
    refuted: int
    not_enough_info: int
    unverifiable: int
    hallucination_rate: float = Field(
        description="refuted / (supported + refuted). Share of checkable claims that are wrong.")
    coverage: float = Field(
        description="(supported + refuted) / total. How much of the text could be checked.")
    risk: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, description="LLM output to check.")
    context: Optional[str] = Field(
        default=None,
        description="Optional source document. If given, claims are checked against it "
                    "(RAG faithfulness mode).")
    sources: Optional[List[Source]] = Field(
        default=None,
        description="Where to look for evidence. Default: only the context if one is given "
                    "(strict faithfulness), otherwise Wikipedia. Pass e.g. "
                    "['wikipedia'] with a context to accept either.")
    top_k: int = Field(default=5, ge=1, le=10)
    include_all_evidence: bool = False


class VerifyResponse(BaseModel):
    summary: Summary
    results: List[ClaimResult]
    elapsed_ms: int
    models: dict


class ClaimsRequest(BaseModel):
    text: str = Field(min_length=1)


class ClaimsResponse(BaseModel):
    claims: List[Claim]


# ============================================================================
# CLAIMS
# ============================================================================
# Turn free text into small, self-contained claims that can each be checked on their own.
#
# Why this matters: an NLI model judges ONE statement against evidence. If a sentence mixes a
# true fact with a false one ("Einstein was a German-born physicist and won 3 Nobel prizes"),
# checking it as a whole gives a muddy answer. So we split it, and every piece keeps its subject:
#
#     "He was born in Ulm in 1879 and died in Princeton."
#         -> "Albert Einstein was born in Ulm in 1879"
#         -> "Albert Einstein died in Princeton"
#
# We use spaCy's dependency parse instead of splitting on the word "and", which breaks noun
# phrases ("Johnson and Johnson") and drops the subject from the second half.


SUBJ_DEPS = {"nsubj", "nsubjpass"}
ARG_DEPS = {"attr", "dobj", "acomp", "oprd"}
SUBJECT_PRONOUNS = {"he", "she", "it", "they"}
REL_PRONOUNS = {"who", "which", "that"}
SEARCHABLE_ENTS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART",
                   "PRODUCT", "LAW", "LANGUAGE"}
OPINION_START = re.compile(
    r"^(i|we|you)\s+(think|believe|feel|guess|hope|would|should|might)\b"
    r"|^(in my opinion|personally|let me|feel free|hope this helps)", re.I)


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        return spacy.load(settings.spacy_model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model '{settings.spacy_model}' missing. Run: "
            f"python -m spacy download {settings.spacy_model}") from exc


# ---------------------------------------------------------------- token helpers
def _subtree(tok, blocked: Set[int]) -> Set[int]:
    """All token indices under `tok`, skipping any branch rooted at a blocked index."""
    out, stack = set(), [tok]
    while stack:
        t = stack.pop()
        if t.i in blocked:
            continue
        out.add(t.i)
        stack.extend(t.children)
    return out


def _text(doc, ids: Iterable[int]) -> str:
    ids = sorted(ids)
    raw = "".join(doc[i].text_with_ws for i in ids)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"\s+([,.;:!?])", r"\1", raw)
    return raw.strip(" ,;:-")


def _noun_phrase(tok) -> Set[int]:
    """The noun phrase headed by `tok`, without its coordinated partners or relative clauses."""
    blocked = {c.i for c in tok.children
               if c.dep_ in ("conj", "cc", "appos", "punct") or _is_extracted_relcl(c)}
    return _subtree(tok, blocked)


def _is_extracted_relcl(tok) -> bool:
    """A 'who/which/that ...' clause. These become their own claims."""
    return tok.dep_ == "relcl" and any(
        c.dep_ in SUBJ_DEPS and c.lower_ in REL_PRONOUNS for c in tok.children)


def _is_checkworthy(sent) -> bool:
    text = sent.text.strip()
    if len(sent) < 4 or text.endswith("?"):
        return False
    if OPINION_START.search(text):
        return False
    # The small spaCy model sometimes tags a verb as a noun ("The Titanic sank"), so a
    # subject is accepted as proof of a clause too.
    return any(t.pos_ in ("VERB", "AUX") or t.dep_ in SUBJ_DEPS for t in sent)


# ---------------------------------------------------------------- main splitter
class ClaimExtractor:
    def __init__(self, nlp=None):
        self.nlp = nlp or get_nlp()

    def extract(self, text: str, max_claims: Optional[int] = None) -> List[Claim]:
        doc = self.nlp(text)
        claims: List[Claim] = []
        seen: Set[str] = set()
        last_subject: Optional[str] = None

        for sent in doc.sents:
            if not _is_checkworthy(sent):
                continue
            pieces, last_subject = self._split_sentence(doc, sent, last_subject)
            for piece_text, ents in pieces:
                key = piece_text.lower()
                if len(piece_text.split()) < 3 or key in seen:
                    continue
                seen.add(key)
                claims.append(Claim(
                    id=len(claims), text=piece_text, sentence=sent.text.strip(),
                    start=sent.start_char, end=sent.end_char, entities=ents))
                if max_claims and len(claims) >= max_claims:
                    return claims
        return claims

    # Each piece is (claim_text, entities_for_search)
    def _split_sentence(self, doc, sent, last_subject):
        root = sent.root
        subj = next((c for c in root.children if c.dep_ in SUBJ_DEPS), None)
        if subj is None:  # fragment or odd parse: keep the sentence whole
            return [(sent.text.strip(), self._entities(doc, set(range(sent.start, sent.end))))], last_subject

        # 1. Work out the subject text(s), resolving a leading pronoun.
        subj_members = [subj] + [c for c in subj.conjuncts if c.i > subj.i]
        subj_texts = []
        for s in subj_members:
            ids = _noun_phrase(s)
            txt = _text(doc, ids)
            if txt.lower() in SUBJECT_PRONOUNS and last_subject:
                txt = last_subject
            subj_texts.append((txt, ids))
        if subj_texts[0][0].lower() not in SUBJECT_PRONOUNS:
            last_subject = subj_texts[0][0]

        subj_block = _subtree(subj, set())
        predicates = [root] + [c for c in root.children
                               if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")]
        # Only "who/which/that" clauses become separate claims. Others ("the first person to
        # walk on the Moon") are part of the meaning and stay inside their claim.
        relcls = [t for t in sent if _is_extracted_relcl(t)]
        pred_idx = {p.i for p in predicates}

        pieces = []
        for p in predicates:
            own_subj = next((c for c in p.children if c.dep_ in SUBJ_DEPS), None)
            if p is not root and own_subj is not None:
                # Independent clause ("X happened, and Y did Z") -> its own subject.
                sub_texts = [(_text(doc, _noun_phrase(own_subj)), _noun_phrase(own_subj))]
                block = _subtree(own_subj, set())
            else:
                sub_texts, block = subj_texts, subj_block

            blocked = set(block) | (pred_idx - {p.i}) | {r.i for r in relcls}
            blocked |= {c.i for c in p.children if c.dep_ == "cc"}
            blocked |= {c.i for c in p.children if c.dep_ == "punct" and c.i == sent.end - 1}
            pred_ids = _subtree(p, blocked)

            for variant in self._split_coordinated_args(p, pred_ids):
                for s_text, s_ids in sub_texts:
                    claim = self._join(doc, s_text, s_ids, variant)
                    pieces.append((claim, self._entities(doc, variant | s_ids, s_text)))

        # 2. Relative clauses: "a physicist who developed relativity" -> "<subject> developed relativity"
        for r in relcls:
            pron = next(c for c in r.children if c.dep_ in SUBJ_DEPS and c.lower_ in REL_PRONOUNS)
            head = r.head
            describes_subject = head.dep_ in ("attr", "appos") or (
                head.dep_ == "conj" and any(a.dep_ == "attr" for a in head.ancestors))
            if describes_subject:
                ref_text, ref_ids = subj_texts[0]
            else:
                ref_ids = _noun_phrase(head)
                ref_text = _text(doc, ref_ids)
            body = _subtree(r, {pron.i})
            claim = self._join(doc, ref_text, ref_ids, body)
            pieces.append((claim, self._entities(doc, body | ref_ids, ref_text)))

        # 3. Appositives: "Einstein, a German physicist, ..." -> "Einstein is a German physicist"
        for ap in (c for c in subj.children if c.dep_ == "appos"):
            ap_ids = _noun_phrase(ap)
            claim = f"{subj_texts[0][0]} is {_text(doc, ap_ids)}"
            pieces.append((claim, self._entities(doc, ap_ids, subj_texts[0][0])))

        return pieces, last_subject

    @staticmethod
    def _split_coordinated_args(pred, pred_ids: Set[int]) -> List[Set[int]]:
        """'was a physicist and a chemist' -> ['was a physicist', 'was a chemist']."""
        arg = next((c for c in pred.children
                    if c.dep_ in ARG_DEPS and c.i in pred_ids and c.conjuncts), None)
        if arg is None:
            return [pred_ids]
        members = [arg] + [m for m in arg.conjuncts if m.i > arg.i]
        arg_full = {i for i in pred_ids if i in {t.i for t in arg.subtree}}
        base = pred_ids - arg_full
        variants = []
        for m in members:
            variants.append(base | (_noun_phrase(m) & pred_ids))
        return variants

    @staticmethod
    def _join(doc, subj_text: str, subj_ids: Set[int], pred_ids: Set[int]) -> str:
        """Glue subject + predicate in sentence order, swapping in a resolved subject."""
        pred_txt = _text(doc, pred_ids)
        if not subj_ids or min(subj_ids) < min(pred_ids, default=0):
            return f"{subj_text} {pred_txt}".strip()
        return f"{pred_txt} {subj_text}".strip()  # inverted order, rare

    @staticmethod
    def _entities(doc, ids: Set[int], subject: Optional[str] = None) -> List[str]:
        out: List[str] = []
        if subject and subject.lower() not in SUBJECT_PRONOUNS:
            subj_ents = [e for e in doc.ents if e.text in subject and e.label_ in SEARCHABLE_ENTS]
            out.extend(e.text for e in subj_ents)
            if not subj_ents:
                out.append(subject)
        for e in doc.ents:
            if e.label_ in SEARCHABLE_ENTS and any(t.i in ids for t in e) and e.text not in out:
                out.append(e.text)
        # drop entries contained in a longer one ("Einstein" when "Albert Einstein" is there)
        return [e for e in out if not any(e != o and e in o for o in out)]


# ============================================================================
# RETRIEVAL
# ============================================================================
# Evidence sources. Each retriever turns a claim into candidate evidence sentences ("passages").
#
# Ranking which passage actually matters happens later (ranking.py), so retrievers can be
# generous: it's fine to return a few hundred sentences from the right Wikipedia article.


log = logging.getLogger("halluscan.retrieval")

WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_SENTENCES_PER_DOC = 300


@dataclass(frozen=True)
class Passage:
    text: str
    source: str               # "wikipedia" | "context" | "web"
    title: Optional[str] = None
    url: Optional[str] = None


class Retriever(Protocol):
    name: str

    def retrieve(self, claim: Claim) -> List[Passage]: ...


# ------------------------------------------------------------------ sentence splitting
@lru_cache(maxsize=1)
def _sentencizer():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    return nlp


def split_sentences(text: str, limit: int = MAX_SENTENCES_PER_DOC) -> List[str]:
    # Wikipedia plain-text extracts carry "== Section ==" headers; they are not evidence.
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("=="))
    out = []
    for s in _sentencizer()(text).sents:
        s = re.sub(r"\s+", " ", s.text).strip()
        if len(s) >= 20:
            out.append(s)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ Wikipedia
class WikipediaRetriever:
    """Uses the official MediaWiki API directly (the old `wikipedia` pip package is unmaintained
    and throws on every disambiguation page)."""

    name = "wikipedia"

    def __init__(self, pages_per_claim: int = settings.wiki_pages_per_claim):
        self.pages_per_claim = pages_per_claim
        self.session = requests.Session()
        self.session.headers["User-Agent"] = settings.user_agent

    def retrieve(self, claim: Claim) -> List[Passage]:
        titles: List[str] = []
        # Entity lookups find the main page ("Albert Einstein"); a full-text search on the claim
        # itself finds pages about the event ("1921 Nobel Prize in Physics").
        for query, limit in [*[(e, 1) for e in claim.entities[:2]], (claim.text, 2)]:
            for t in self._search(query, limit):
                if t not in titles:
                    titles.append(t)
        passages: List[Passage] = []
        for title in titles[: self.pages_per_claim]:
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            for sent in self._page_sentences(title):
                passages.append(Passage(sent, "wikipedia", title, url))
        return passages

    @lru_cache(maxsize=2048)
    def _search(self, query: str, limit: int) -> tuple:
        try:
            r = self.session.get(WIKI_API, timeout=settings.http_timeout, params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": limit, "format": "json", "srprop": ""})
            r.raise_for_status()
            return tuple(hit["title"] for hit in r.json()["query"]["search"])
        except Exception as exc:  # network errors should degrade, not crash the request
            log.warning("wikipedia search failed for %r: %s", query, exc)
            return ()

    @lru_cache(maxsize=512)
    def _page_sentences(self, title: str) -> tuple:
        try:
            r = self.session.get(WIKI_API, timeout=settings.http_timeout, params={
                "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                "titles": title, "format": "json"})
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            extract = next(iter(pages.values())).get("extract", "")
            return tuple(split_sentences(extract[:120_000]))
        except Exception as exc:
            log.warning("wikipedia fetch failed for %r: %s", title, exc)
            return ()


# ------------------------------------------------------------------ user-supplied context
class ContextRetriever:
    """RAG faithfulness mode: the only allowed evidence is the document the user gives us."""

    name = "context"

    def __init__(self, context: str):
        self._passages = [Passage(s, "context", "provided context")
                          for s in split_sentences(context, limit=2000)]

    def retrieve(self, claim: Claim) -> List[Passage]:
        return list(self._passages)


# ------------------------------------------------------------------ open web (optional)
class WebRetriever:
    """Search-engine snippets via the `ddgs` package. Off by default: it is rate-limited and
    snippets are lower quality than full articles."""

    name = "web"

    def retrieve(self, claim: Claim) -> List[Passage]:
        try:
            from ddgs import DDGS
        except ImportError:
            log.warning("web source requested but `ddgs` is not installed")
            return []
        try:
            hits = DDGS().text(claim.text, max_results=5)
        except Exception as exc:
            log.warning("web search failed: %s", exc)
            return []
        out = []
        for h in hits:
            for s in split_sentences(h.get("body", ""), limit=5):
                out.append(Passage(s, "web", h.get("title"), h.get("href")))
        return out


# ============================================================================
# MODELS
# ============================================================================
# The two neural pieces: an embedder (to find relevant evidence) and an NLI cross-encoder
# (to decide whether that evidence supports or contradicts the claim).
#
# Both are loaded once and reused for every request. Loading BART-large on every call was the
# single biggest slowdown in the original notebook.


log = logging.getLogger("halluscan.models")


def _device() -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ------------------------------------------------------------------ NLI
class NLIScorer(Protocol):
    name: str

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """pairs = [(premise/evidence, hypothesis/claim)] -> [{entailment, neutral, contradiction}]"""


class HFNLIScorer:
    def __init__(self, model_name: str = settings.nli_model, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self.torch = torch
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        # Different checkpoints order their labels differently. Read the mapping, don't assume.
        self.label_index = {}
        for idx, label in self.model.config.id2label.items():
            l = label.lower()
            key = ("entailment" if l.startswith("entail") else
                   "contradiction" if l.startswith("contra") else "neutral")
            self.label_index[key] = int(idx)
        if set(self.label_index) != {"entailment", "neutral", "contradiction"}:
            raise ValueError(f"{model_name} is not a 3-way NLI model: {self.model.config.id2label}")

    def predict(self, pairs):
        out: List[Dict[str, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                                 truncation="only_first", max_length=512,
                                 padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1).cpu().tolist()
            for row in probs:
                out.append({k: row[v] for k, v in self.label_index.items()})
        return out


# ------------------------------------------------------------------ Embeddings / relevance
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of in on at to for from by with and or but is was were are be been
being it its he she they his her their this that which who whom as into than then also""".split())


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class Relevance(Protocol):
    name: str

    def scores(self, queries: Sequence[str], docs: Sequence[str]) -> List[List[float]]:
        """Similarity matrix [len(queries)][len(docs)] in 0..1"""


class EmbeddingRelevance:
    def __init__(self, model_name: str = settings.embed_model):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=_device())

    def scores(self, queries, docs):
        if not docs:
            return [[] for _ in queries]
        q = self.model.encode(list(queries), normalize_embeddings=True, batch_size=64)
        d = self.model.encode(list(docs), normalize_embeddings=True, batch_size=64)
        return (q @ d.T).clip(0, 1).tolist()


class LexicalRelevance:
    """No-model fallback: BM25-style overlap squashed into 0..1. Good enough to find the
    sentence that mentions the same names and numbers."""

    name = "lexical-bm25"

    def scores(self, queries, docs):
        doc_toks = [_tokens(d) for d in docs]
        n = max(len(docs), 1)
        df = Counter(t for toks in doc_toks for t in set(toks))
        avg_len = sum(map(len, doc_toks)) / n if docs else 1
        out = []
        for q in queries:
            q_toks = set(_tokens(q))
            row = []
            for toks in doc_toks:
                tf = Counter(toks)
                s = 0.0
                for t in q_toks:
                    if t in tf:
                        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                        s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(toks) / avg_len))
                row.append(s)
            top = max(row, default=0) or 1
            row = [r / top * min(1.0, len(q_toks & set(t)) / max(len(q_toks), 1) * 2)
                   for r, t in zip(row, doc_toks)]
            out.append(row)
        return out


@lru_cache(maxsize=1)
def default_nli() -> NLIScorer:
    return HFNLIScorer()


@lru_cache(maxsize=1)
def default_relevance() -> Relevance:
    try:
        return EmbeddingRelevance()
    except Exception as exc:
        log.warning("embedding model unavailable (%s); using lexical ranking", exc)
        return LexicalRelevance()


# ============================================================================
# PIPELINE
# ============================================================================
# The full check:  text -> claims -> evidence -> rank -> NLI -> verdict per claim -> summary.
#
#         claim ──► retrievers ──► ~hundreds of sentences
#                                     │  relevance (embeddings)
#                                     ▼
#                               top-k sentences  (+1 "combined" premise of the top 3)
#                                     │  NLI cross-encoder, one batch for the whole request
#                                     ▼
#                      entailment / contradiction / neutral per sentence
#                                     │  aggregate (max support vs max refute)
#                                     ▼
#                 SUPPORTED | REFUTED | NOT_ENOUGH_INFO | UNVERIFIABLE


_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


class Detector:
    def __init__(self, extractor: Optional[ClaimExtractor] = None,
                 nli: Optional[NLIScorer] = None,
                 relevance: Optional[Relevance] = None,
                 retrievers: Optional[Dict[str, Retriever]] = None):
        self._extractor = extractor
        self._nli = nli
        self._relevance = relevance
        self._retrievers = retrievers

    # Lazy so that importing the package (or running tests) never downloads a model.
    @property
    def extractor(self):
        self._extractor = self._extractor or ClaimExtractor()
        return self._extractor

    @property
    def nli(self):
        self._nli = self._nli or default_nli()
        return self._nli

    @property
    def relevance(self):
        self._relevance = self._relevance or default_relevance()
        return self._relevance

    @property
    def retrievers(self):
        if self._retrievers is None:
            self._retrievers = {"wikipedia": WikipediaRetriever(), "web": WebRetriever()}
        return self._retrievers

    def warm_up(self):
        _ = self.extractor, self.nli, self.relevance

    def model_info(self) -> dict:
        return {"nli": getattr(self._nli, "name", None),
                "relevance": getattr(self._relevance, "name", None),
                "claims": settings.spacy_model}

    # ------------------------------------------------------------------ public API
    def extract_claims(self, text: str) -> List[Claim]:
        return self.extractor.extract(text, max_claims=settings.max_claims)

    def verify(self, text: str, context: Optional[str] = None,
               sources: Optional[Sequence[str]] = None, top_k: int = settings.evidence_top_k,
               include_all_evidence: bool = False) -> VerifyResponse:
        t0 = time.perf_counter()
        claims = self.extract_claims(text)

        if sources is None:
            sources = [] if context else ["wikipedia"]
        active: List[Retriever] = []
        if context:
            active.append(ContextRetriever(context))
        for s in sources:
            if s != "context" and s in self.retrievers:
                active.append(self.retrievers[s])

        # 1) Retrieval is network-bound, so do all claims in parallel.
        with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
            candidates = list(pool.map(lambda c: self._gather(c, active), claims))

        # 2) Rank: one embedding pass over every unique sentence in this request.
        ranked = self._rank(claims, candidates, top_k)

        # 3) NLI: one batched pass over every (evidence, claim) pair.
        pairs, owners = [], []
        for ci, (claim, items) in enumerate(zip(claims, ranked)):
            for p, rel in items:
                pairs.append((p.text, claim.text))
                owners.append((ci, p, rel))
        probs = self.nli.predict(pairs) if pairs else []

        per_claim: List[List[Evidence]] = [[] for _ in claims]
        for (owner, passage, rel), pr in zip(owners, probs):
            per_claim[owner].append(Evidence(
                text=passage.text, source=passage.source, title=passage.title,
                url=passage.url, relevance=round(rel, 4),
                entailment=round(pr["entailment"], 4),
                contradiction=round(pr["contradiction"], 4),
                neutral=round(pr["neutral"], 4)))

        results = [self._decide(c, ev, bool(context), include_all_evidence)
                   for c, ev in zip(claims, per_claim)]
        return VerifyResponse(summary=summarize(results, context_mode=bool(context)), results=results,
                              elapsed_ms=int((time.perf_counter() - t0) * 1000),
                              models=self.model_info())

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _gather(claim: Claim, retrievers: Iterable[Retriever]) -> List[Passage]:
        seen, out = set(), []
        for r in retrievers:
            for p in r.retrieve(claim):
                if p.text not in seen:
                    seen.add(p.text)
                    out.append(p)
        return out

    def _rank(self, claims, candidates, top_k):
        uniq: Dict[str, int] = {}
        for cands in candidates:
            for p in cands:
                uniq.setdefault(p.text, len(uniq))
        docs = list(uniq)
        sims = self.relevance.scores([c.text for c in claims], docs) if docs else [[] for _ in claims]

        ranked = []
        for claim, cands, row in zip(claims, candidates, sims):
            scored = sorted(((p, row[uniq[p.text]]) for p in cands), key=lambda x: -x[1])
            top = [(p, s) for p, s in scored[:top_k] if s >= settings.min_relevance]
            # Some facts are spread over two sentences ("He moved to Princeton. He died there.")
            # so we also test the top three together as a single premise.
            if len(top) >= 2:
                joined = " ".join(p.text for p, _ in top[:3])
                first = top[0][0]
                top.append((Passage(joined, first.source, f"{first.title or ''} (combined)".strip(),
                                    first.url), top[0][1]))
            ranked.append(top)
        return ranked

    @staticmethod
    def _decide(claim: Claim, evidence: List[Evidence], context_mode: bool,
                include_all: bool) -> ClaimResult:
        flags: List[str] = []
        if not evidence:
            if context_mode:
                flags.append("not_found_in_context")
            return ClaimResult(claim=claim, verdict=Verdict.UNVERIFIABLE, confidence=0.0,
                               flags=flags)

        best_support = max(evidence, key=lambda e: e.entailment)
        best_refute = max(evidence, key=lambda e: e.contradiction)
        sup, ref = best_support.entailment, best_refute.contradiction

        if sup >= settings.entail_threshold and sup >= ref:
            verdict, conf, best = Verdict.SUPPORTED, sup, best_support
        elif ref >= settings.contra_threshold:
            verdict, conf, best = Verdict.REFUTED, ref, best_refute
        else:
            verdict, conf = Verdict.NOT_ENOUGH_INFO, 1 - max(sup, ref)
            best = max(evidence, key=lambda e: e.relevance)

        # Numbers are where small NLI models slip most ("born in 1880" vs "born in 1879").
        # If the most relevant sentence has years and none match the claim's, say so.
        claim_years = set(_YEAR.findall(claim.text))
        top_rel = max(evidence, key=lambda e: e.relevance)
        ev_years = set(_YEAR.findall(top_rel.text))
        if claim_years and ev_years and not (claim_years & ev_years) and verdict != Verdict.SUPPORTED:
            flags.append("numeric_mismatch")
            if verdict == Verdict.NOT_ENOUGH_INFO and top_rel.relevance >= 0.6:
                verdict, conf, best = Verdict.REFUTED, round(0.5 + top_rel.relevance / 4, 4), top_rel
                flags.append("refuted_by_numeric_rule")

        if context_mode and verdict != Verdict.SUPPORTED:
            flags.append("not_grounded_in_context")
        if best.relevance < 0.4:
            flags.append("weak_evidence_match")

        evidence_sorted = sorted(evidence, key=lambda e: -max(e.entailment, e.contradiction))
        return ClaimResult(claim=claim, verdict=verdict, confidence=round(conf, 4),
                           best_evidence=best,
                           evidence=evidence_sorted if include_all else evidence_sorted[:3],
                           flags=flags)


def summarize(results: List[ClaimResult], context_mode: bool = False) -> Summary:
    count = {v: sum(r.verdict == v for r in results) for v in Verdict}
    s, r = count[Verdict.SUPPORTED], count[Verdict.REFUTED]
    checkable, total = s + r, len(results)
    if context_mode:
        # Faithfulness: the document is the whole truth. Anything it doesn't back up is
        # a hallucination, whether it is contradicted or just not there.
        r, checkable = total - s, total
    rate = r / checkable if checkable else 0.0
    if checkable == 0:
        risk = "UNKNOWN"
    elif r == 0:
        risk = "LOW"
    elif rate <= 0.34:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    return Summary(total_claims=total, supported=s, refuted=r,
                   not_enough_info=count[Verdict.NOT_ENOUGH_INFO],
                   unverifiable=count[Verdict.UNVERIFIABLE],
                   hallucination_rate=round(rate, 4),
                   coverage=round(checkable / total, 4) if total else 0.0, risk=risk)


# ============================================================================
# LOADERS
# ============================================================================
# Read uploaded files into plain text.


class UnsupportedFile(ValueError):
    pass


def text_from_upload(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFile("Supported file types: .pdf, .docx, .txt, .md")


# ============================================================================
# API
# ============================================================================
# HTTP API.
#
# Run:   uvicorn halluscan.api:app --port 8000
# Docs:  http://localhost:8000/docs   (interactive Swagger UI, generated automatically)
# Demo:  http://localhost:8000/


log = logging.getLogger("halluscan.api")
_detector = Detector()


def get_detector() -> Detector:
    """FastAPI dependency. Tests swap this for a detector with fake models."""
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.preload_models:
        log.info("loading models...")
        await run_in_threadpool(_detector.warm_up)
        log.info("models ready: %s", _detector.model_info())
    yield


app = FastAPI(
    title="HalluScan",
    version=__version__,
    description="Detects hallucinations in LLM output by splitting it into atomic claims and "
                "checking each one against Wikipedia, the web, or a document you provide.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _check_size(text: str, field: str = "text"):
    if len(text) > settings.max_chars:
        raise HTTPException(413, f"`{field}` is {len(text)} chars; limit is {settings.max_chars}.")


@app.get("/", include_in_schema=False)
def demo_page():
    return HTMLResponse(DEMO_HTML)


@app.get("/health")
def health(detector: Detector = Depends(get_detector)):
    return {"status": "ok", "version": __version__, "models": detector.model_info()}


@app.post("/v1/claims", response_model=ClaimsResponse,
          summary="Only split text into atomic claims (fast, no evidence lookup)")
async def claims(req: ClaimsRequest, detector: Detector = Depends(get_detector)):
    _check_size(req.text)
    return ClaimsResponse(claims=await run_in_threadpool(detector.extract_claims, req.text))


@app.post("/v1/verify", response_model=VerifyResponse,
          summary="Check every claim in a piece of text")
async def verify(req: VerifyRequest, detector: Detector = Depends(get_detector)):
    _check_size(req.text)
    if req.context:
        _check_size(req.context, "context")
    # Model inference is blocking; keep the event loop free for other requests.
    return await run_in_threadpool(
        detector.verify, req.text, req.context, req.sources, req.top_k, req.include_all_evidence)


@app.post("/v1/verify/file", response_model=VerifyResponse,
          summary="Same as /v1/verify but the text comes from a PDF, DOCX, TXT or MD file")
async def verify_file(file: UploadFile = File(...),
                      context: Optional[str] = Form(None),
                      sources: Optional[List[str]] = Form(None),
                      top_k: int = Form(5),
                      detector: Detector = Depends(get_detector)):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB.")
    try:
        text = text_from_upload(file.filename, data)
    except UnsupportedFile as exc:
        raise HTTPException(415, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Could not read file: {exc}")
    if not text.strip():
        raise HTTPException(422, "No text found in the file (scanned PDFs need OCR first).")
    _check_size(text)
    bad = [s for s in (sources or []) if s not in ("wikipedia", "web", "context")]
    if bad:
        raise HTTPException(422, f"Unknown source(s): {bad}")
    return await run_in_threadpool(detector.verify, text, context, sources, max(1, min(top_k, 10)))


# ============================================================================
# DEMO PAGE (served at "/")
# ============================================================================
DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HalluScan</title>
<style>
  :root {
    --bg: #f7f6f2; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6a65; --line: #e3e1da;
    --ok: #1f7a4d; --ok-bg: #e3f3ea; --bad: #b3261e; --bad-bg: #fbe5e3;
    --meh: #8a6100; --meh-bg: #fbf0d6; --none: #5f5e5a; --none-bg: #eeede8; --accent: #1d1d1b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #151514; --panel: #1f1f1d; --ink: #ecebe6; --muted: #a09f99; --line: #34332f;
      --ok: #6fd3a0; --ok-bg: #1b3327; --bad: #ff8f86; --bad-bg: #3a1e1c;
      --meh: #f0c766; --meh-bg: #3a2f14; --none: #b8b7b1; --none-bg: #2a2a27; --accent: #ecebe6;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.02em; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  label { font-weight: 600; font-size: 13px; display: block; margin: 16px 0 6px; }
  textarea { width: 100%; min-height: 120px; padding: 12px; border: 1px solid var(--line);
             border-radius: 10px; background: var(--panel); color: var(--ink); font: inherit; resize: vertical; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 14px; }
  button { background: var(--accent); color: var(--bg); border: 0; border-radius: 10px;
           padding: 10px 18px; font: inherit; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  .chk { display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 14px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 28px 0 18px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
  .stat b { display: block; font-size: 22px; }
  .stat span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .claim { font-weight: 600; }
  .tag { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .05em;
         padding: 2px 8px; border-radius: 999px; margin-right: 8px; vertical-align: 2px; }
  .SUPPORTED { color: var(--ok); background: var(--ok-bg); }
  .REFUTED { color: var(--bad); background: var(--bad-bg); }
  .NOT_ENOUGH_INFO { color: var(--meh); background: var(--meh-bg); }
  .UNVERIFIABLE { color: var(--none); background: var(--none-bg); }
  .ev { color: var(--muted); font-size: 14px; margin-top: 8px; border-left: 3px solid var(--line); padding-left: 10px; }
  .ev a { color: inherit; }
  .meta { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .err { color: var(--bad); margin-top: 16px; }
</style>
</head>
<body>
<main>
  <h1>HalluScan</h1>
  <p class="sub">Paste LLM output. Each claim is checked against Wikipedia, or against your own source text.</p>

  <label for="text">LLM output</label>
  <textarea id="text">Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He won the Nobel Prize in Physics in 1925 and died in Princeton.</textarea>

  <label for="ctx">Source document (optional, switches to faithfulness mode)</label>
  <textarea id="ctx" placeholder="Paste the document the answer should be grounded in..." style="min-height:70px"></textarea>

  <div class="row">
    <button id="go">Check claims</button>
    <span class="chk"><input type="checkbox" id="wiki" checked><label for="wiki" style="margin:0;font-weight:400">Wikipedia</label></span>
    <span class="chk"><input type="checkbox" id="web"><label for="web" style="margin:0;font-weight:400">Web search</label></span>
    <span id="status" class="chk"></span>
  </div>
  <div id="err" class="err"></div>
  <div id="out"></div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = x => Math.round(x * 100) + "%";

$("go").onclick = async () => {
  const sources = [];
  if ($("wiki").checked) sources.push("wikipedia");
  if ($("web").checked) sources.push("web");
  const body = { text: $("text").value };
  // With a source document, check strictly against it; otherwise use the ticked sources.
  if ($("ctx").value.trim()) body.context = $("ctx").value;
  else body.sources = sources;
  $("go").disabled = true; $("err").textContent = ""; $("status").textContent = "Checking...";
  try {
    const r = await fetch("/v1/verify", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    render(data);
    $("status").textContent = `${data.elapsed_ms} ms`;
  } catch (e) {
    $("err").textContent = e.message; $("status").textContent = "";
  } finally { $("go").disabled = false; }
};

function render(d) {
  const s = d.summary;
  let html = `<div class="stats">
    <div class="stat"><span>Risk</span><b>${s.risk}</b></div>
    <div class="stat"><span>Hallucination rate</span><b>${pct(s.hallucination_rate)}</b></div>
    <div class="stat"><span>Coverage</span><b>${pct(s.coverage)}</b></div>
    <div class="stat"><span>Supported / refuted</span><b>${s.supported} / ${s.refuted}</b></div>
  </div>`;
  for (const r of d.results) {
    const ev = r.best_evidence;
    html += `<div class="card">
      <div><span class="tag ${r.verdict}">${r.verdict.replaceAll("_", " ")}</span><span class="claim">${esc(r.claim.text)}</span></div>
      ${ev ? `<div class="ev">${esc(ev.text)}<br>${ev.url ? `<a href="${esc(ev.url)}" target="_blank" rel="noopener">${esc(ev.title)}</a>` : esc(ev.title || ev.source)}</div>` : ""}
      <div class="meta">confidence ${pct(r.confidence)}${r.flags.length ? " · " + r.flags.map(esc).join(", ") : ""}</div>
    </div>`;
  }
  $("out").innerHTML = html;
}
</script>
</body>
</html>
"""


# ============================================================================
# COMMAND LINE
# ============================================================================
def _print_report(resp: VerifyResponse) -> None:
    icons = {"SUPPORTED": "[ OK ]", "REFUTED": "[FAIL]", "NOT_ENOUGH_INFO": "[ ?? ]",
             "UNVERIFIABLE": "[ -- ]"}
    s = resp.summary
    print(f"\nRisk: {s.risk}   hallucination rate: {s.hallucination_rate:.0%}   "
          f"coverage: {s.coverage:.0%}   claims: {s.total_claims}   ({resp.elapsed_ms} ms)\n")
    for r in resp.results:
        print(f"{icons[r.verdict.value]} {r.verdict.value:<15} {r.claim.text}  (conf {r.confidence:.2f})")
        if r.best_evidence:
            ev = r.best_evidence
            print(f"         evidence: {ev.text[:150]}{'...' if len(ev.text) > 150 else ''}")
            print(f"         source:   {ev.url or ev.title or ev.source}")
        if r.flags:
            print(f"         flags:    {', '.join(r.flags)}")
    print()


def main(argv=None) -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="HalluScan: detect hallucinations in LLM output.")
    ap.add_argument("text", nargs="?", help="Text to check (or use --file, or pipe via stdin)")
    ap.add_argument("--file", help="Read the text from a .pdf, .docx, .txt or .md file")
    ap.add_argument("--context", help="Source document file for faithfulness mode")
    ap.add_argument("--sources", nargs="*", choices=["wikipedia", "web"],
                    help="Evidence sources (default: wikipedia, or only the context if given)")
    ap.add_argument("--top-k", type=int, default=settings.evidence_top_k)
    ap.add_argument("--json", action="store_true", help="Print the full JSON report")
    ap.add_argument("--serve", action="store_true", help="Run the HTTP API + demo page")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.serve:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.file:
        with open(args.file, "rb") as f:
            text = text_from_upload(args.file, f.read())
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.error("give some text, --file, or --serve")

    context = None
    if args.context:
        with open(args.context, "rb") as f:
            context = text_from_upload(args.context, f.read())

    logging.basicConfig(level=logging.WARNING)
    resp = Detector().verify(text, context=context, sources=args.sources, top_k=args.top_k)
    if args.json:
        print(resp.model_dump_json(indent=2))
    else:
        _print_report(resp)


if __name__ == "__main__":
    main()
