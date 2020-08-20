""" Testing the inference benchmark code. """

import pathlib
import unittest

from inference import benchmark_inference


class BenchmarkInference(unittest.TestCase):
    def test_classifier(self) -> None:
        benchmark_inference.benchmark(
            "test_classifier",
            batch_size=1,
            model_type="classifier",
            run_time=5.0,
            models_dir=pathlib.Path("external"),
        )


if __name__ == "__main__":
    unittest.main()
