Tests
==============================================================================

Testing is mainly done with Bazel. For python code, please use ``unittest`` and
``doctests`` where applicable.

To run all test, use a recursive search with Bazel (``...``):
::

    bazel test //...

Here, the ``//`` refers to the folder where our ``WORKSPACE`` file resides.

To run a specific test, find the ``BUILD`` file that specifies the test target:
::

    bazel test //third_party/detectron2:anchors_doctest

Testing Style
---------------------------

We prefer to write tests next to the file being tested. This helps test remain short
and readable. For example if you have a file ``module/foo.py`` please write the tests
in ``module/foo_test.py``. You'll need to place the proper test target in the
corresponding ``BUILD`` file.

It is important to have robust tests for `hawk_eye/inference` code and the related
dependencies because this is the bulk of our distributed package. Other items like
model training and data generation are harder and less important to test thoroughly.
