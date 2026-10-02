"""Checksum failures are tested against tiny authored files, never model weights."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_local_nli.py"
SPEC = importlib.util.spec_from_file_location("fetch_local_nli", SCRIPT)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


class NLIFetchTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)
        siblings = []
        for name in fetcher.FILES:
            value = ("synthetic-" + name).encode()
            (path / name).write_bytes(value)
            lfs = SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()) if name == "model.safetensors" else None
            siblings.append(SimpleNamespace(rfilename=name, size=len(value), lfs=lfs))
        return path, siblings

    def test_all_sizes_and_weight_lfs_digest_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path, siblings = self.fixture(directory)
            verified = fetcher.verify_files(path, siblings)
            self.assertEqual(len(verified), len(fetcher.FILES))
            self.assertTrue(next(item for item in verified if item["name"] == "model.safetensors")["hf_lfs_sha256"])

    def test_same_size_weight_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, siblings = self.fixture(directory)
            weight = path / "model.safetensors"
            weight.write_bytes(b"x" * weight.stat().st_size)
            with self.assertRaisesRegex(ValueError, "lfs_checksum_mismatch"):
                fetcher.verify_files(path, siblings)

    def test_wrong_size_and_missing_weight_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, siblings = self.fixture(directory)
            (path / "config.json").write_text("wrong size")
            with self.assertRaisesRegex(ValueError, "wrong_size"):
                fetcher.verify_files(path, siblings)
            path, siblings = self.fixture(directory)
            next(item for item in siblings if item.rfilename == "model.safetensors").lfs = None
            with self.assertRaisesRegex(ValueError, "weight_lfs_checksum_missing"):
                fetcher.verify_files(path, siblings)

    def test_unsafe_repo_python_is_not_allowlisted(self):
        self.assertEqual(len(fetcher.FILES), 8)
        self.assertFalse(any(name.endswith((".py", ".bin")) for name in fetcher.FILES))


if __name__ == "__main__":
    unittest.main()
