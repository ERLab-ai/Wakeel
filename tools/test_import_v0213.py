"""Importer unit tests: failures must be explicit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("wakeel_import", Path(__file__).with_name("import_v0213.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestImport(unittest.TestCase):
    def test_safe_paths(self):
        for path in ("../secret", "/secret", r"a\b", "a//b", "a/../b", "C:/abc"):
            self.assertFalse(m.safe_path(path), path)
        self.assertTrue(m.safe_path("content/chapter-09/results.json"))
    def test_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "wrong.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("README.md", "not canonical")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                m.verify(f)
    def test_existing_directory_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "wrong.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("README.md", "not canonical")
            output = Path(tmp) / "dest"
            output.mkdir()
            (output / "existing").write_text("safe")
            with self.assertRaises(ValueError):
                m.extract(f, output)
            self.assertEqual((output / "existing").read_text(), "safe")

if __name__ == "__main__":
    unittest.main()
