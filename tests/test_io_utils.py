import hashlib
import tempfile
import unittest
from pathlib import Path

from calibration_tool.io_utils import sha256_file


class IoUtilsTests(unittest.TestCase):
    def test_normalized_hash_ignores_windows_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"a\nb\n")
            crlf.write_bytes(b"a\r\nb\r\n")
            self.assertEqual(sha256_file(lf), sha256_file(crlf))
            self.assertNotEqual(
                sha256_file(lf, normalize_newlines=False),
                sha256_file(crlf, normalize_newlines=False),
            )

    def test_streaming_hash_handles_crlf_across_chunk_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            data = b"x" * (1024 * 1024 - 1) + b"\r\nend\r"
            path.write_bytes(data)
            expected = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
            self.assertEqual(sha256_file(path), expected)
            self.assertEqual(
                sha256_file(path, normalize_newlines=False),
                hashlib.sha256(data).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
