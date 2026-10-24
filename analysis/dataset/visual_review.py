"""Fixed preprint visual nominations and deferred independent-review helpers."""
from collections import Counter
import csv
from datetime import date
from pathlib import Path
import re


CATEGORIES = ("retain", "exclude", "uncertain")
FIELDS = ("question_id", "OF_rating", "OF_reason", "OF_date", "DG_rating", "DG_reason", "DG_date",
          "MO_rating", "MO_reason", "MO_date", "final_decision")
# Fixed agent nomination pool: 107 screened items plus ambiguous 000795, 2026-10-08.
# Vault source_register.md owns screening provenance; review_packet.py exports these IDs.
CANDIDATE_IDS = tuple("medqa-us-evaluation-c3b905ccfa66152d-" + suffix for suffix in (
    "000010 000012 000019 000032 000046 000075 000078 000096 000112 000117 000124 000127 "
    "000140 000141 000147 000161 000206 000221 000233 000234 000252 000257 000265 000271 "
    "000275 000285 000312 000331 000332 000337 000341 000366 000374 000398 000407 000412 "
    "000449 000455 000468 000470 000471 000476 000485 000486 000519 000522 000538 000547 "
    "000558 000573 000576 000582 000588 000604 000617 000623 000634 000641 000645 000654 "
    "000658 000669 000677 000683 000684 000686 000710 000712 000717 000726 000728 000731 "
    "000740 000762 000786 000794 000795 000822 000831 000836 000862 000882 000911 000928 "
    "000936 000946 000973 000978 001003 001021 001033 001061 001065 001080 001096 001111 "
    "001144 001164 001178 001187 001198 001202 001203 001212 001238 001240 001249 001260 "
).split())


def fixed_visual_exclusions(manifest):
    """SAP v1.1: exclude the fixed 108 nominations only in a descriptive sensitivity."""
    items = [row["item_id"] for row in manifest["items"]]
    if (len(items) != 1273 or len(set(items)) != 1273
            or len(CANDIDATE_IDS) != 108 or len(set(CANDIDATE_IDS)) != 108
            or not set(CANDIDATE_IDS) <= set(items)):
        raise ValueError("Fixed visual sensitivity requires 1,273 unique questions and the exact 108 nominations")
    return {"policy": "fixed_nominated_108", "post_hoc": True,
            "description": "Questions flagged for potentially missing visual material",
            "primary_n": 1273, "candidate_n": 108, "retained_n": 1165,
            "excluded_ids": sorted(CANDIDATE_IDS), "study_id": manifest["study_id"],
            "questions_sha256": manifest["bindings"]["questions"]}


def review_agreement(rows):
    """Original independent ratings only; nominal, unweighted Cohen kappa."""
    table = [[0]*3 for _ in range(3)]
    missing, counts = Counter(), {who: Counter() for who in ("OF", "DG")}
    for row in rows:
        a, b = row.get("OF_rating", ""), row.get("DG_rating", "")
        for who, value in (("OF", a), ("DG", b)):
            if not value:
                missing[who] += 1
            elif value not in CATEGORIES:
                raise ValueError("Unknown independent review category")
            else:
                counts[who][value] += 1
        if a and b:
            table[CATEGORIES.index(a)][CATEGORIES.index(b)] += 1
    n = sum(map(sum, table))
    of = [sum(r) for r in table]
    dg = [sum(table[i][j] for i in range(3)) for j in range(3)]
    agree = sum(table[i][i] for i in range(3))
    expected_numerator = sum(a*b for a, b in zip(of, dg))
    kappa = ((n*agree-expected_numerator)/(n*n-expected_numerator)
             if n and expected_numerator != n*n else None)
    return {"categories": list(CATEGORIES), "table": table, "paired_n": n,
            "OF_counts": {c: counts["OF"][c] for c in CATEGORIES},
            "DG_counts": {c: counts["DG"][c] for c in CATEGORIES},
            "paired_OF_counts": dict(zip(CATEGORIES, of)), "paired_DG_counts": dict(zip(CATEGORIES, dg)),
            "missing_ratings": {who: missing[who] for who in ("OF", "DG")},
            "agreement_n": agree, "percentage_agreement": 100*agree/n if n else None,
            "both_uncertain_n": table[2][2], "kappa": kappa,
            "kappa_reason": None if kappa is not None else ("expected agreement is 1" if n else "no paired ratings")}


def validate_review_rows(rows, expected_ids=CANDIDATE_IDS):
    """Complete pool, preserved ratings, dated reasons and consistent adjudication."""
    ids = [r.get("question_id") for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Review must contain every candidate exactly once, with no extra IDs")
    for row in rows:
        if not set(FIELDS) <= set(row):
            raise ValueError("Review CSV is missing required reviewer/adjudicator columns")
        for who in ("OF", "DG"):
            if row[who+"_rating"] not in CATEGORIES or not row[who+"_reason"].strip():
                raise ValueError("Complete independent ratings and reasons required")
            _date(row[who+"_date"])
        a, b = row["OF_rating"], row["DG_rating"]
        adjudicate = a != b or a == "uncertain"
        final = row["final_decision"]
        if final not in ("retain", "exclude"):
            raise ValueError("Unresolved final decision")
        if adjudicate:
            if row["MO_rating"] != final or not row["MO_reason"].strip():
                raise ValueError("MO's definite adjudication and reason must match final_decision")
            _date(row["MO_date"])
        elif final != a or any(row[k] for k in ("MO_rating", "MO_reason", "MO_date")):
            raise ValueError("Concordant definite ratings require their consensus and blank MO fields")
    return {"excluded_ids": sorted(r["question_id"] for r in rows if r["final_decision"] == "exclude"),
            "irr": review_agreement(rows)}


def _date(value):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("Review dates must use YYYY-MM-DD")
    date.fromisoformat(value)


def load_visual_review(path, expected_sha256, manifest):
    """Only an independently approved checksum authorizes this additional input."""
    path = Path(path)
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Set the approved final visual-review CSV SHA-256")
    payload = path.read_bytes()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Visual-review CSV differs from its approved SHA-256")
    import io
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
        raise ValueError("Duplicate CSV column names")
    rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV row")
    result = validate_review_rows(rows)
    if not set(CANDIDATE_IDS) <= {q["item_id"] for q in manifest["items"]}:
        raise ValueError("Visual-review pool does not match the frozen question identities")
    return {**result, "rows": rows, "input_sha256": expected_sha256,
            "questions_sha256": manifest["bindings"]["questions"],
            "study_id": manifest["study_id"], "candidate_n": len(rows)}
