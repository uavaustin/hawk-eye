#!/usr/bin/env python3

import doctest
import unittest

import third_party


def test_third_party() -> None:
    test_suite = unittest.TestSuite()
    test_suite.addTests(doctest.DocTestSuite("third_party.vovnet.vovnet"))
    return test_suite


suite = test_third_party()
runner = unittest.TextTestRunner()
runner.run(suite)
