"""Read-only view: retain novel text and identify exact reused word spans.

Only prior documents with recorded complete first readings may be reused.
Numbers and words are not changed. This is not a semantic reviewer or a labeler.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def spans(text):
    return list(re.finditer(r"\S+", text))


def display(source_path, review_path, sample, start, end):
    sources = json.loads(Path(source_path).read_text())
    prior = json.loads(Path(review_path).read_text())["documents"]
    eligible = {d["sample_number"] for d in prior
                if d["coverage"]["full_pdf_semantically_read"] and not d["coverage"]["unreviewed_pages"]}
    index = defaultdict(list)
    prior_pages = {}
    width, minimum = 12, 40
    for doc in sources:
        num = doc["source"]["sample_number"]
        if num not in eligible or num == sample:
            continue
        for page in doc["pages"]:
            tokens = tuple(m.group() for m in spans(page["text"]))
            identity = (num, page["page"])
            prior_pages[identity] = tokens
            for pos in range(len(tokens) - width + 1):
                key = tokens[pos:pos + width]
                if len(index[key]) < 20:
                    index[key].append((identity, pos))
    doc = next(d for d in sources if d["source"]["sample_number"] == sample)
    for page in doc["pages"][start - 1:end]:
        text = page["text"]
        matches = spans(text)
        tokens = tuple(m.group() for m in matches)
        i = printed = 0
        print(f"\nPAGE {page['page']} OCR={page['ocr_used']} unresolved={page.get('selected_ocr_unresolved', False)}")
        while i < len(tokens):
            best = (0, None, None)
            for identity, pos in index.get(tokens[i:i + width], []):
                old = prior_pages[identity]
                length = width
                while i + length < len(tokens) and pos + length < len(old) and tokens[i + length] == old[pos + length]:
                    length += 1
                if length > best[0]:
                    best = (length, identity, pos)
            length, identity, pos = best
            if length >= minimum:
                print(text[printed:matches[i].start()])
                print(f"[EXACT REUSE: sample {identity[0]}, page {identity[1]}, words {pos}:{pos+length}, {length} words]")
                i += length
                printed = matches[i-1].end()
            else:
                i += 1
        print(text[printed:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--source", default="artifacts/experience_holdout_source_view_v1/source_documents.json")
    parser.add_argument("--review", default="artifacts/experience_holdout_review_progress_v23/compiled_source_reviews.json")
    args = parser.parse_args()
    display(args.source, args.review, args.sample, args.start, args.end)
