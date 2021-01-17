#!/usr/bin/env python3

"""Tests covering the inference functions."""

import doctest
import unittest

from hawk_eye.inference import benchmark_inference
from hawk_eye.inference import find_targets


def test_inference_doctests() -> None:
    test_suite = unittest.TestSuite()
    test_suite.addTests(doctest.DocTestSuite("hawk_eye.inference"))
    return test_suite


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

    suite = test_inference_doctests()
    runner = unittest.TextTestRunner(failfast=True)
    out = runner.run(suite)
    assert len(out.failures) == 0, "Failures in test"

    unittest.main()
