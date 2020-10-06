#!/usr/bin/env python3

""" Tests covering the inference functions. """

import doctest
import unittest

import inference


def test_inference_doctests() -> None:
    test_suite = unittest.TestSuite()
    test_suite.addTests(doctest.DocTestSuite("inference.find_targets"))
    return test_suite


if __name__ == "__main__":

    suite = test_inference_doctests()
    runner = unittest.TextTestRunner(failfast=True)
    out = runner.run(suite)
    assert len(out.failures) == 0, "Failures in test"

    unittest.main()
