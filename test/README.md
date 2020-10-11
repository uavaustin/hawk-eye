## Notes on testing with Bazel
> A growing collection on tips to help write Python unit tests.

#### Running an Individual Test
`bazel test //test:test_benchmark_inference`

#### Seeing Test Output
`py_test` bazel targets are output supressed by default, but to see the
test output, which can be helpful for test debugging, can be done like so:

`bazel test --test_output=streamed //test:test_benchmark_inference`

#### Test Structure
Each test file corresponds to a module in `hawk_eye`, and within each file,
you'll find the tests corresponding to files and functions.
