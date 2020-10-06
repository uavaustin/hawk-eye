#!/usr/bin/env python3

import doctest
import unittest

import third_party


def test_third_party() -> None:
    test_suite = unittest.TestSuite()
    test_suite.addTests(doctest.DocTestSuite("third_party.vovnet.vovnet"))
    return test_suite


if __name__ == "__main__":
    suite = test_third_party()
    runner = unittest.TextTestRunner(failfast=True)
    out = runner.run(suite)
    assert len(out.failures) == 0, "Failures in test"
