import tempfile
import unittest
from pathlib import Path

import compare_documents as driver


class StubProcessor:
    def __init__(self):
        self.last_args = None

    def compare(self, text1, text2, normalize, include_alignments, top_k):
        self.last_args = {
            "text1": text1,
            "text2": text2,
            "normalize": normalize,
            "include_alignments": include_alignments,
            "top_k": top_k,
        }
        return {
            "cosine_similarity": 0.5,
            "interpretation": "medium",
        }


class DriverTests(unittest.TestCase):
    def test_compare_pair_reads_file_and_passes_text_to_processor(self):
        processor = StubProcessor()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "doc1.txt"
            file_path.write_text("file content", encoding="utf-8")

            result = driver.compare_pair(
                doc1_value=str(file_path),
                doc2_value="raw text",
                processor=processor,
                normalize=True,
                include_alignments=False,
                top_k=3,
            )

        self.assertEqual(result["interpretation"], "medium")
        self.assertEqual(processor.last_args["text1"], "file content")
        self.assertEqual(processor.last_args["text2"], "raw text")
        self.assertTrue(processor.last_args["normalize"])
        self.assertFalse(processor.last_args["include_alignments"])
        self.assertEqual(processor.last_args["top_k"], 3)

    def test_infer_batch_format_uses_extension_when_unspecified(self):
        jsonl = driver.infer_batch_format(Path("out.jsonl"), None)
        csv = driver.infer_batch_format(Path("out.csv"), None)

        self.assertEqual(jsonl, "jsonl")
        self.assertEqual(csv, "csv")


if __name__ == "__main__":
    unittest.main()
