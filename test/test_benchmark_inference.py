""" Testing the inference benchmark code. """

import pathlib
import unittest

from inference import benchmark_inference
from inference import find_targets


class BenchmarkInference(unittest.TestCase):
    def test_classifier(self) -> None:
        benchmark_inference.benchmark(
            timestamp=find_targets._PROD_MODELS["clf"],
            batch_size=1,
            model_type="classifier",
            run_time=2.5,
        )

    def test_detector(self) -> None:
        benchmark_inference.benchmark(
            timestamp=find_targets._PROD_MODELS["det"],
            batch_size=1,
            model_type="detector",
            run_time=2.5,
        )


if __name__ == "__main__":
    unittest.main()
