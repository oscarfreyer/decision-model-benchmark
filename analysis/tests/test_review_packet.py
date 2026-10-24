"""Manual exporter checks using fictional questions; never read research inputs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.dataset import review_packet


class ReviewPacket(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.ids = ("fictional-02", "fictional-01")
        self.questions = [{"item_id": item, "stem": item + ' "question", line 1\nline 2: α\u2028end',
                           "options": [{"label": label, "text": item + label + ', "choice"\nα'}
                                       for label in "ABCD"]} for item in reversed(self.ids)]
        self.bind_questions(self.questions)
        self.patch_constant("CANDIDATE_IDS", self.ids)
        self.patch_constant("QUESTION_COUNT", 2)

    def patch_constant(self, name, value):
        context = patch.object(review_packet, name, value)
        context.start()
        self.addCleanup(context.stop)

    def bind_questions(self, questions):
        payload = ("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n").encode()
        (self.input / "questions.jsonl").write_bytes(payload)
        question_hash = hashlib.sha256(payload).hexdigest()
        manifest = json.dumps({"files": [{"path": "questions.jsonl", "bytes": len(payload),
                                           "sha256": question_hash}]}).encode()
        (self.input / "input_manifest.json").write_bytes(manifest)
        self.patch_constant("QUESTIONS_SHA256", question_hash)
        self.patch_constant("INPUT_MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())

    def test_exact_text_fixed_order_and_one_blank_reviewer_column(self):
        payload = review_packet.packet_bytes(self.input, "DG")
        rows = list(csv.reader(io.StringIO(payload.decode(), newline="")))
        self.assertEqual(rows[0], ["question_id", "question", "option_A", "option_B",
                                  "option_C", "option_D", "reviewer_DG"])
        self.assertEqual([row[0] for row in rows[1:]], list(self.ids))
        expected = next(q for q in self.questions if q["item_id"] == self.ids[0])
        self.assertEqual(rows[1][1], expected["stem"])
        self.assertEqual(rows[1][2:6], [option["text"] for option in expected["options"]])
        self.assertTrue(all(len(row) == 7 and row[-1] == "" for row in rows[1:]))
        self.assertTrue(payload.startswith(b'"question_id","question"'))
        self.assertTrue(payload.endswith(b',""\r\n'))

    def test_verify_is_read_only_and_export_refuses_overwrite_or_input_destination(self):
        output = self.root / "packet.csv"
        review_packet.build_review_packet(self.input, output=output)
        original = output.read_bytes()
        result = review_packet.build_review_packet(self.input, verify_csv=output)
        self.assertTrue(result["verified_existing"])
        self.assertEqual(output.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            review_packet.build_review_packet(self.input, output=output)
        with self.assertRaisesRegex(ValueError, "immutable"):
            review_packet.build_review_packet(self.input, output=self.input / "packet.csv")
        output.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            review_packet.build_review_packet(self.input, verify_csv=output)

    def test_changed_question_or_inventory_bytes_fail_before_output(self):
        output = self.root / "packet.csv"
        for filename in ("questions.jsonl", "input_manifest.json"):
            path = self.input / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, r"\bdiffers?\b"):
                review_packet.build_review_packet(self.input, output=output)
            self.assertFalse(output.exists())
            path.write_bytes(original)

    def test_missing_duplicate_candidates_and_reordered_options_fail(self):
        for ids in (("absent", "fictional-01"), ("fictional-01", "fictional-01")):
            with patch.object(review_packet, "CANDIDATE_IDS", ids):
                with self.assertRaisesRegex(ValueError, "Candidate inventory"):
                    review_packet.packet_bytes(self.input)
        self.questions[0]["options"].reverse()
        self.bind_questions(self.questions)
        with self.assertRaisesRegex(ValueError, "option order"):
            review_packet.packet_bytes(self.input)


if __name__ == "__main__":
    unittest.main()
