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
