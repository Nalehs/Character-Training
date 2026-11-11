"""Crash-safe JSONL output: a stage that dies must not leave a file that looks finished."""
import tempfile
import unittest
from pathlib import Path

from character_pipeline.data.io import partial_path, read_jsonl, resumable_jsonl


class ResumableJsonl(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = Path(self.dir.name) / "sub" / "out.jsonl"

    def test_crash_leaves_only_the_partial_and_resume_finishes_it(self):
        with self.assertRaises(RuntimeError):
            with resumable_jsonl(self.out) as (done, write):
                self.assertEqual(done, [])
                write({"n": 0})
                raise RuntimeError("server went away")
        self.assertFalse(self.out.exists())
        with partial_path(self.out).open("a") as f:
            f.write('{"n": 1, "torn')  # a row cut off mid-write
        with resumable_jsonl(self.out) as (done, write):
            self.assertEqual(done, [{"n": 0}])
            write({"n": 1})
        self.assertEqual(read_jsonl(self.out), [{"n": 0}, {"n": 1}])
        self.assertFalse(partial_path(self.out).exists())

    def test_final_file_is_never_overwritten(self):
        with resumable_jsonl(self.out) as (_, write):
            write({"n": 0})
        with self.assertRaises(SystemExit):
            with resumable_jsonl(self.out):
                pass
        self.assertEqual(read_jsonl(self.out), [{"n": 0}])

    def test_no_resume_discards_the_partial(self):
        partial_path(self.out).parent.mkdir(parents=True)
        partial_path(self.out).write_text('{"old": 1}\n')
        with resumable_jsonl(self.out, resume=False) as (done, write):
            self.assertEqual(done, [])
            write({"n": 0})
        self.assertEqual(read_jsonl(self.out), [{"n": 0}])

    def test_read_jsonl_skips_blank_lines_and_names_bad_ones(self):
        p = Path(self.dir.name) / "x.jsonl"
        p.write_text('{"a": 1}\n\n  \n{"a": 2}\n')
        self.assertEqual(read_jsonl(p), [{"a": 1}, {"a": 2}])
        p.write_text('{"a": 1}\nnope\n')
        with self.assertRaisesRegex(ValueError, "x.jsonl:2"):
            read_jsonl(p)


if __name__ == "__main__":
    unittest.main()
