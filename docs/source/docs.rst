Documentation
================================

The documentation uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_
to auto-generate the `documentation website
<https://uavaustin.github.io/hawk-eye/index.html>`_.


Building Locally
--------------------------------
The docs are built by going into the ``docs/source`` folder and running

::

    sphinx-build -b html . _build

You can then look inside ``docs/source/_build`` and open up ``index.html``
to view your local changes.


Remote Publishing
-------------------------------
The docs are automatically built by the Github Actions workflow defined by
``.github/workflows/docs_build.yml``. The docs will be published to our
repositorie's Github Pages site whenever changes are merged into the main
branch.
