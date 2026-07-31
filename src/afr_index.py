"""Inverted index over the AFR news corpus.

A naive regex scan of the 219,538-article corpus takes ~15 seconds, which alone
would breach the 60-second response budget. This module builds a doc-level
inverted index so pattern counts resolve in milliseconds.

Exactness is preserved by construction:

  1. the index proposes a **superset** of candidate documents for a pattern;
  2. the caller's regex is then run against those candidates to decide matches;
  3. patterns that cannot be decomposed fall back to a full corpus scan.

The tokenizer is ``\\w+``, which is exactly Python's ``\\b`` word-boundary
alphabet, so ``\\bQBE\\b`` is equivalent to "the token ``qbe`` occurs" and needs
no verification pass at all.

Search scope is fixed by the challenge rules: HEADLINE + SUBHEAD + INTRO + TEXT
concatenated, case-insensitive, counted once per record.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import get_settings

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LITERAL_RE = re.compile(r"^\w+$", re.UNICODE)


def _normalise_headline(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()

#: Beyond this share of the corpus, intersecting postings is slower than scanning.
_CANDIDATE_SCAN_THRESHOLD = 0.40


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def _iter_articles(afr_dir: Path) -> Iterator[tuple[int, int, dict]]:
    files = sorted(p for p in afr_dir.glob("AFR_*.jsonl"))
    for file_id, path in enumerate(files):
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if line:
                    yield file_id, line_no, json.loads(line)


def blob_of(record: dict) -> str:
    """The searchable text: the four content fields, concatenated and lowercased."""
    parts = (
        record.get("HEADLINE") or "",
        record.get("SUBHEAD") or "",
        record.get("INTRO") or "",
        record.get("TEXT") or "",
    )
    return " ".join(parts).replace("\n", " ").replace("\r", " ").lower()


def build(verbose: bool = True) -> Path:
    """Build the index artifacts. Takes a couple of minutes; run once."""
    settings = get_settings()
    out_dir = settings.artifacts_dir / "afr"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in settings.afr_dir.glob("AFR_*.jsonl"))
    offsets: list[int] = []
    lengths: list[int] = []
    dates: list[int] = []
    locators: list[tuple[int, int]] = []
    headlines: list[str] = []
    postings: dict[str, list[int]] = {}

    position = 0
    with open(out_dir / "corpus.bin", "wb") as corpus:
        for doc_id, (file_id, line_no, record) in enumerate(_iter_articles(settings.afr_dir)):
            blob = blob_of(record)
            encoded = blob.encode("utf-8")
            corpus.write(encoded)

            offsets.append(position)
            lengths.append(len(encoded))
            position += len(encoded)

            raw_date = str(record.get("PUBLICATIONDATE") or "0")[:8]
            dates.append(int(raw_date) if raw_date.isdigit() else 0)
            locators.append((file_id, line_no))
            headlines.append((record.get("HEADLINE") or "").replace("\n", " ").replace("\r", " "))

            for token in set(TOKEN_RE.findall(blob)):
                postings.setdefault(token, []).append(doc_id)

            if verbose and doc_id % 20000 == 0 and doc_id:
                print(f"  indexed {doc_id:,} articles", file=sys.stderr)

    vocab = sorted(postings)
    posting_offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    flat = np.empty(sum(len(v) for v in postings.values()), dtype=np.uint32)
    cursor = 0
    for i, token in enumerate(vocab):
        ids = postings[token]
        posting_offsets[i] = cursor
        flat[cursor : cursor + len(ids)] = ids
        cursor += len(ids)
    posting_offsets[-1] = cursor

    np.savez(
        out_dir / "meta.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int32),
        dates=np.asarray(dates, dtype=np.int32),
        file_ids=np.asarray([f for f, _ in locators], dtype=np.int16),
        line_nos=np.asarray([l for _, l in locators], dtype=np.int32),
    )
    np.savez(out_dir / "postings.npz", flat=flat, offsets=posting_offsets)
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    (out_dir / "headlines.txt").write_text("\n".join(headlines), encoding="utf-8")
    (out_dir / "files.json").write_text(json.dumps(files), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({"documents": len(offsets), "vocabulary": len(vocab), "postings": int(cursor)}),
        encoding="utf-8",
    )
    if verbose:
        print(
            f"built AFR index: {len(offsets):,} documents, {len(vocab):,} tokens, "
            f"{cursor:,} postings -> {out_dir}",
            file=sys.stderr,
        )
    return out_dir


# --------------------------------------------------------------------------
# pattern decomposition
# --------------------------------------------------------------------------
def split_alternatives(pattern: str) -> list[str]:
    """Split on top-level ``|`` only, ignoring escapes, groups and classes."""
    parts, depth_paren, depth_class, buffer, i = [], 0, 0, [], 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            buffer.append(pattern[i : i + 2])
            i += 2
            continue
        if char == "[":
            depth_class += 1
        elif char == "]" and depth_class:
            depth_class -= 1
        elif char == "(" and not depth_class:
            depth_paren += 1
        elif char == ")" and not depth_class and depth_paren:
            depth_paren -= 1
        elif char == "|" and not depth_paren and not depth_class:
            parts.append("".join(buffer))
            buffer = []
            i += 1
            continue
        buffer.append(char)
        i += 1
    parts.append("".join(buffer))
    return [p for p in (part.strip() for part in parts) if p]


def _strip_boundaries(part: str) -> str:
    while part.startswith(("\\b", "^")):
        part = part[2:] if part.startswith("\\b") else part[1:]
    while part.endswith("\\b") or part.endswith("$"):
        part = part[:-2] if part.endswith("\\b") else part[:-1]
    return part


@dataclass(frozen=True)
class _Alternative:
    parts: list[str]
    token_exact: bool
    """True only for a single plain word wrapped in ``\\b`` on both sides.

    Such an alternative is equivalent to "this token occurs", so its postings
    are the answer and no verification pass is needed. Anything else - phrases,
    quantifiers, unanchored literals - yields candidates that must be verified,
    because ``cash rate`` also matches the token ``rates`` and ``interest``
    also matches inside ``disinterest``.
    """


_METACHARS = set(".*+?{}[]()|^$\\")


def literal_prefix(alternative: str) -> str | None:
    """Longest leading literal every match of ``alternative`` must contain.

    Used as a fast ``in`` pre-filter before the real regex runs. A trailing
    optional character is dropped, so ``interest rates?`` yields
    ``interest rate`` and never excludes a genuine match.
    """
    literal: list[str] = []
    i = 0
    while i < len(alternative):
        char = alternative[i]
        if char == "\\" and alternative[i : i + 2] == "\\b":
            i += 2
            continue
        if char in ("?", "*"):
            if literal:
                literal.pop()  # the preceding character was optional
            break
        if char in _METACHARS:
            break
        literal.append(char)
        i += 1
    text = "".join(literal).strip().lower()
    return text if len(text) >= 3 else None


def _analyse(alternative: str) -> _Alternative | None:
    fully_bounded = alternative.startswith("\\b") and alternative.endswith("\\b")
    inner = _strip_boundaries(alternative)
    if not inner:
        return None
    raw_parts = [p for p in inner.split() if p]
    if not raw_parts:
        return None
    parts = [_strip_boundaries(p) or p for p in raw_parts]
    exact = fully_bounded and len(parts) == 1 and bool(_LITERAL_RE.match(parts[0]))
    return _Alternative(parts=parts, token_exact=exact)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------
class AfrIndex:
    """Read-only after construction, so it is safe to share across requests.

    The single corpus file handle is guarded by a lock because ``seek``+``read``
    is not atomic and the agent serves concurrent queries.
    """

    def __init__(self, directory: Path):
        self.dir = directory
        self._read_lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._search_cache: dict[str, np.ndarray] = {}
        self._part_cache: dict[str, np.ndarray] = {}
        self._normalised_headlines: list[str] | None = None
        meta = np.load(directory / "meta.npz")
        self.offsets = meta["offsets"]
        self.lengths = meta["lengths"]
        self.dates = meta["dates"]
        self.file_ids = meta["file_ids"]
        self.line_nos = meta["line_nos"]

        postings = np.load(directory / "postings.npz")
        self.postings = postings["flat"]
        self.posting_offsets = postings["offsets"]

        self.vocab = (directory / "vocab.txt").read_text(encoding="utf-8").split("\n")
        self.vocab_index = {token: i for i, token in enumerate(self.vocab)}
        self.headlines = (directory / "headlines.txt").read_text(encoding="utf-8").split("\n")
        self.files = json.loads((directory / "files.json").read_text(encoding="utf-8"))
        self._corpus_path = directory / "corpus.bin"
        self._handle = open(self._corpus_path, "rb")

        # One newline-joined string so a vocabulary lookup is a single regex
        # pass over ~3.5 MB instead of 374k individual Python-level searches.
        self._vocab_blob = "\n".join(self.vocab)
        starts, cursor = [], 0
        for token in self.vocab:
            starts.append(cursor)
            cursor += len(token) + 1
        self._vocab_starts = np.asarray(starts, dtype=np.int64)

    # -- storage ---------------------------------------------------------
    @property
    def document_count(self) -> int:
        return int(self.offsets.shape[0])

    def doc_text(self, doc_id: int) -> str:
        with self._read_lock:
            self._handle.seek(int(self.offsets[doc_id]))
            raw = self._handle.read(int(self.lengths[doc_id]))
        return raw.decode("utf-8", "replace")

    def normalised_headlines(self) -> list[str]:
        """Lowercased, accent-stripped headlines, computed once."""
        if self._normalised_headlines is None:
            with self._search_lock:
                if self._normalised_headlines is None:
                    self._normalised_headlines = [_normalise_headline(h) for h in self.headlines]
        return self._normalised_headlines

    def source_record(self, doc_id: int) -> dict:
        """Re-read the original (correctly cased) JSONL record."""
        path = get_settings().afr_dir / self.files[int(self.file_ids[doc_id])]
        target = int(self.line_nos[doc_id])
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no == target:
                    return json.loads(line)
        raise KeyError(f"document {doc_id} not found in {path}")

    # -- postings --------------------------------------------------------
    def _token_docs(self, token: str) -> np.ndarray:
        i = self.vocab_index.get(token)
        if i is None:
            return np.empty(0, dtype=np.uint32)
        return self.postings[self.posting_offsets[i] : self.posting_offsets[i + 1]]

    def _substring_docs(self, part: str, anchor: str = "any") -> np.ndarray | None:
        """Docs holding any token in which ``part`` can match.

        A part free of whitespace can only match inside a single token, so
        matching it against the vocabulary yields every document the part could
        possibly hit - never fewer.

        ``anchor`` exploits the spaces inside a multi-word pattern. In
        ``interest rates?`` the space forces a token boundary, so ``interest``
        can only match at the *end* of a token and ``rates?`` only at the
        *start* of the next one. Without that, ``rates?`` also matches
        "corporate" and "moderate" and the candidate set explodes.
        """
        key = f"{anchor}:{part}"
        cached = self._part_cache.get(key)
        if cached is not None:
            return cached
        expression = {
            "any": part,
            "start": f"^(?:{part})",
            "end": f"(?:{part})$",
            "both": f"^(?:{part})$",
        }[anchor]
        try:
            matcher = re.compile(expression, re.IGNORECASE | re.MULTILINE)
        except re.error:
            return None

        token_ids = set()
        for match in matcher.finditer(self._vocab_blob):
            if "\n" in match.group(0):
                continue  # a match spanning tokens is not a token match
            token_ids.add(int(np.searchsorted(self._vocab_starts, match.start(), side="right") - 1))
        hits = [self._token_docs(self.vocab[i]) for i in token_ids]
        docs = np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=np.uint32)
        self._part_cache[key] = docs
        return docs

    def candidates(self, pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Split a pattern into (confirmed docs, docs needing verification).

        Confirmed docs come from ``\\b``-anchored single words, where postings
        are already the exact answer. Everything else is a superset that the
        caller must verify with the real regex. Returns None when the pattern
        cannot be decomposed and a full scan is required.
        """
        alternatives = split_alternatives(pattern)
        if not alternatives:
            return None

        confirmed: list[np.ndarray] = []
        unverified: list[np.ndarray] = []
        for alternative in alternatives:
            analysed = _analyse(alternative)
            if analysed is None:
                return None

            if analysed.token_exact:
                confirmed.append(self._token_docs(analysed.parts[0].lower()))
                continue

            parts = analysed.parts
            docs: np.ndarray | None = None
            for position, part in enumerate(parts):
                if len(parts) == 1:
                    anchor = "any"
                elif position == 0:
                    anchor = "end"
                elif position == len(parts) - 1:
                    anchor = "start"
                else:
                    anchor = "both"
                part_docs = self._substring_docs(part, anchor)
                if part_docs is None:
                    return None
                docs = part_docs if docs is None else np.intersect1d(docs, part_docs)
                if docs.size == 0:
                    break
            unverified.append(docs if docs is not None else np.empty(0, dtype=np.uint32))

        def _merge(arrays: list[np.ndarray]) -> np.ndarray:
            return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.uint32)

        confirmed_docs = _merge(confirmed)
        pending = _merge(unverified)
        if confirmed_docs.size:
            pending = pending[~np.isin(pending, confirmed_docs)]
        return confirmed_docs, pending

    # -- search ----------------------------------------------------------
    def full_scan(self, matcher: re.Pattern[str]) -> np.ndarray:
        """Exhaustive fallback: scan every document once."""
        hits = [doc_id for doc_id in range(self.document_count) if matcher.search(self.doc_text(doc_id))]
        return np.asarray(hits, dtype=np.uint32)

    def search(self, pattern: str) -> np.ndarray:
        """Doc ids matching ``pattern`` case-insensitively, once per document."""
        cached = self._search_cache.get(pattern)
        if cached is not None:
            return cached

        matcher = re.compile(pattern, re.IGNORECASE)
        split = self.candidates(pattern)

        if split is None:
            result = self.full_scan(matcher)
        else:
            confirmed, pending = split
            if pending.size > self.document_count * _CANDIDATE_SCAN_THRESHOLD:
                result = self.full_scan(matcher)
            else:
                # Substring screening is an order of magnitude cheaper than the
                # alternation regex, and rejects most candidates outright.
                literals = [literal_prefix(a) for a in split_alternatives(pattern)]
                screen = [lit for lit in literals if lit] if all(literals) else []
                verified = []
                for doc_id in pending:
                    text = self.doc_text(int(doc_id))
                    if screen and not any(lit in text for lit in screen):
                        continue
                    if matcher.search(text):
                        verified.append(int(doc_id))
                result = np.union1d(confirmed, np.asarray(verified, dtype=np.uint32)).astype(np.uint32)

        with self._search_lock:
            if len(self._search_cache) > 64:
                self._search_cache.clear()
            self._search_cache[pattern] = result
        return result


_INDEX: AfrIndex | None = None


def index_dir() -> Path:
    return get_settings().artifacts_dir / "afr"


def is_built() -> bool:
    directory = index_dir()
    return all(
        (directory / name).exists()
        for name in ("corpus.bin", "meta.npz", "postings.npz", "vocab.txt", "headlines.txt")
    )


def ensure_built(verbose: bool = True) -> None:
    if not is_built():
        build(verbose=verbose)


def get_index() -> AfrIndex:
    global _INDEX
    if _INDEX is None:
        ensure_built()
        _INDEX = AfrIndex(index_dir())
    return _INDEX
