#!/usr/bin/env python3

import time
import pathlib
import tempfile
import unittest

from hawk_eye.core import asset_manager


class FileUpload(unittest.TestCase):
    def test_file_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)
            tmp_file = tmp_dir / f"{time.perf_counter()}.txt"
            tmp_file.write_text("test")
            asset_manager.upload_file(tmp_file, f"test-output/{tmp_file.name}")

            bucket = asset_manager._get_client_bucket()
            blob = bucket.blob(f"test-output/{tmp_file.name}")

            self.assertTrue(blob.exists())

    def test_no_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            asset_manager.upload_file(
                pathlib.Path(tempfile.NamedTemporaryFile().name), "test-output/none"
            )


if __name__ == "__main__":
    unittest.main()
