## Notes on testing with Bazel
> A growing collection on tips to help write unit tests.

#### Running an Individual Test
`bazel test //test:test_inference`

An individual test can also be run as a python executable like so:
`PYTHONPATH=. test/test_inference.py`

#### Seeing Test Output
`py_test` bazel targets are output supressed by default, but to see the
test output, which can be helpful for test debugging, can be done like so:

`bazel test --test_output=streamed //test:test_inference`

#### Test Structure
For readability, each test file corresponds to a module in `hawk_eye`, and within
each file,
you'll find the tests corresponding to files and functions. This format is not a strict
requirement, and if you feel there is a better way, please reach out.

Refer to the
[`unittest`](https://docs.python.org/3.8/library/unittest.html) docs for information on
testing. Please also use [`doctest`](https://docs.python.org/3/library/doctest.html)
strings where applicable.
