"""Reconstruct the nominated question/option CSV from frozen, answer-free input.

The fixed IDs in visual_review.py encode the agent's 8 October 2026 content
screen (107 items plus ambiguous 000795). This exporter does not screen text.
It is a retained reconstruction of the earlier ad hoc export, not its original
execution record. Run manually from the repository root with python -m.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from analysis.dataset.visual_review import CANDIDATE_IDS


INPUT_MANIFEST_SHA256 = "7cb5399e8d4c4d330fb79a958c33d9d1950e3025525d132de8cb79de78644fa3"
QUESTIONS_SHA256 = "25d1b1bdd767caa1f90a38c5d09f94ad9b29e9942b1301cbca0a707e61b4d7a5"
QUESTION_COUNT = 1273
CONTENT_FIELDS = ("question_id", "question", "option_A", "option_B", "option_C", "option_D")


def packet_bytes(input_dir, reviewer="XX"):
    """Read only the accepted inventory and questions; preserve text exactly."""
    if not isinstance(reviewer, str) or not re.fullmatch(r"[A-Z]{2,8}", reviewer):
        raise ValueError("Reviewer initials must contain 2–8 uppercase letters")
    input_dir = Path(input_dir)
    manifest_bytes = (input_dir / "input_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != INPUT_MANIFEST_SHA256:
        raise ValueError("Input inventory differs from the accepted SHA-256")
    manifest = json.loads(manifest_bytes)
    entries = [entry for entry in manifest["files"] if entry["path"] == "questions.jsonl"]
    questions_bytes = (input_dir / "questions.jsonl").read_bytes()
    if (len(entries) != 1 or entries[0]["sha256"] != QUESTIONS_SHA256
            or entries[0]["bytes"] != len(questions_bytes)
            or hashlib.sha256(questions_bytes).hexdigest() != QUESTIONS_SHA256):
        raise ValueError("Questions differ from the accepted inventory entry")
    questions = [json.loads(line) for line in io.StringIO(questions_bytes.decode("utf-8"))]
    ids = [question["item_id"] for question in questions]
    if len(ids) != QUESTION_COUNT or len(ids) != len(set(ids)):
        raise ValueError("Questions require the full, unique frozen split")
    if len(CANDIDATE_IDS) != len(set(CANDIDATE_IDS)) or not set(CANDIDATE_IDS) <= set(ids):
        raise ValueError("Candidate inventory is duplicated or missing from the questions")
    lookup = {question["item_id"]: question for question in questions}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow((*CONTENT_FIELDS, "reviewer_" + reviewer))
    for item_id in CANDIDATE_IDS:
        question = lookup[item_id]
        options = question["options"]
        if (not isinstance(question["stem"], str) or not isinstance(options, list)
                or [option["label"] for option in options] != ["A", "B", "C", "D"]
                or not all(isinstance(option["text"], str) for option in options)):
            raise ValueError("Candidate requires a text stem and original A–D option order")
        writer.writerow((item_id, question["stem"], *(option["text"] for option in options), ""))
    return stream.getvalue().encode("utf-8")


def build_review_packet(input_dir, *, output=None, verify_csv=None, reviewer="XX"):
    """Write a fresh packet or verify an existing blank packet byte for byte."""
    if (output is None) == (verify_csv is None):
        raise ValueError("Choose exactly one of output or verify_csv")
    if output is not None and Path(output).resolve().is_relative_to(Path(input_dir).resolve()):
        raise ValueError("Review packets must be outside the immutable input folder")
    payload = packet_bytes(input_dir, reviewer)
    if verify_csv is not None:
        if Path(verify_csv).read_bytes() != payload:
            raise ValueError("CSV differs from the reconstructed blank review packet")
        destination = Path(verify_csv)
    else:
        destination = Path(output)
        with destination.open("xb") as stream:
            stream.write(payload)
    return {"generator": "analysis/dataset/review_packet.py", "version": "1.0",
            "input_manifest_sha256": INPUT_MANIFEST_SHA256,
            "questions_sha256": QUESTIONS_SHA256, "candidate_n": len(CANDIDATE_IDS),
            "reviewer_column": "reviewer_" + reviewer,
            "csv_sha256": hashlib.sha256(payload).hexdigest(),
            "csv": str(destination), "verified_existing": verify_csv is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1] / "input")
    parser.add_argument("--reviewer", default="XX", help="Initials used in the blank rating column, e.g. OF or DG")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="Fresh CSV path; parent must exist; never overwrite")
    target.add_argument("--verify-csv", type=Path, help="Verify a blank packet without writing")
    args = parser.parse_args()
    try:
        result = build_review_packet(args.input_dir, output=args.output,
                                     verify_csv=args.verify_csv, reviewer=args.reviewer)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
