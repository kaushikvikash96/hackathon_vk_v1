"""Label AFR articles with market sentiment and direction, using the supplied Qwen.

    python training/label_sentiment.py --count 300

Run this on the cluster, where the LiteLLM `agent-brain` alias is reachable.
Labels are distilled from the organizer-supplied Qwen model - an approved local
service - into the Nemotron synthesis adapter. No external service is used.

Output: training/data/sentiment_labels.jsonl, consumed by make_dataset.py.

Sentiment is the one part of the answer that is a judgement rather than a
computation, so these labels are the only ones in the training set that are not
derived arithmetically. Review a sample by hand before training and record what
you checked in MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import afr_index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.data_rba import rate_on  # noqa: E402
from src.util import fmt_rate, parse_date  # noqa: E402

SCOPES = [
    "the broad ASX", "ASX banking shares", "ASX mining shares", "ASX energy shares",
    "ASX travel shares", "ASX property shares", "rate-sensitive shares",
]

PROMPT = """Read this Australian Financial Review article and classify it.

HEADLINE: {headline}
PUBLISHED: {published}
RBA CASH-RATE TARGET IN FORCE: {rate}

ARTICLE:
{body}

Answer with exactly two lines and nothing else:
SENTIMENT: positive | negative | mixed
DIRECTION: upward | downward | mixed-to-down | mixed-to-up | flat"""

_SENTIMENT_RE = re.compile(r"SENTIMENT:\s*([a-z\- ]+)", re.I)
_DIRECTION_RE = re.compile(r"DIRECTION:\s*([a-z\- ]+)", re.I)

#: Only label articles that plausibly carry market signal.
MARKET_PATTERN = (
    r"\bASX\b|\bshares?\b|\bstocks?\b|\bmarket\b|\binvestors?\b|\bRBA\b|"
    r"\bcash rate\b|\bprofit\b|\bdividend\b"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--out", default="training/data/sentiment_labels.jsonl")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    settings = get_settings()
    try:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_key,
            timeout=httpx.Timeout(60.0, connect=5.0),
            max_retries=1,
        )
    except Exception as exc:
        print(f"cannot reach the labelling model: {exc}", file=sys.stderr)
        return 1

    index = afr_index.get_index()
    candidates = index.search(MARKET_PATTERN)
    print(f"{len(candidates):,} market-relevant articles; sampling {args.count}")

    rng = random.Random(args.seed)
    sample = rng.sample(list(candidates), k=min(args.count, len(candidates)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as handle:
        for position, doc_id in enumerate(sample, start=1):
            record = index.source_record(int(doc_id))
            headline = (record.get("HEADLINE") or "").strip()
            published = str(record.get("PUBLICATIONDATE") or "")
            if len(published) != 8 or not headline:
                continue
            iso_date = f"{published[:4]}-{published[4:6]}-{published[6:8]}"

            decision = rate_on(parse_date(iso_date))
            if decision is None:
                continue
            rate = fmt_rate(decision.target)
            body = " ".join(filter(None, [record.get("INTRO") or "", record.get("TEXT") or ""]))[:4000]

            try:
                reply = client.chat.completions.create(
                    model=settings.brain_model,
                    messages=[{"role": "user", "content": PROMPT.format(
                        headline=headline, published=iso_date, rate=rate, body=body)}],
                    temperature=0.0,
                    max_tokens=30,
                )
                text = reply.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [{position}] labelling failed: {type(exc).__name__}", file=sys.stderr)
                continue

            sentiment_match = _SENTIMENT_RE.search(text)
            direction_match = _DIRECTION_RE.search(text)
            if not sentiment_match or not direction_match:
                continue

            sentiment = sentiment_match.group(1).strip().lower()
            direction = direction_match.group(1).strip().lower()
            if sentiment not in {"positive", "negative", "mixed"}:
                continue

            handle.write(json.dumps({
                "doc_id": int(doc_id),
                "headline": headline,
                "publication_date": iso_date,
                "rate": rate,
                "sentiment": sentiment,
                "direction": direction,
                "scope": rng.choice(SCOPES),
            }, ensure_ascii=False) + "\n")
            written += 1
            if position % 25 == 0:
                print(f"  labelled {written}/{position}")

    print(f"wrote {written} labels -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
