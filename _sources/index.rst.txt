.. hawk-eye documentation master file, created by
   sphinx-quickstart on Wed Jan 13 19:35:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hawk-eye
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


An Application to Detect Ground-Based Targets

Structure
~~~~~~~~~~~~~~~~~~~~~~
- **configs** - different configuration files for model training.
- **core** - a shared library accross different applications.
- **data_generation** - scripts for genertion datasets.
- **inference** - code for performing inference.
- **train** - scripts related to training models.


.. toctree::
    :caption: Setup
    :maxdepth: 2

    setup/setup

.. toctree::
    :caption: Core
    :maxdepth: 2

    core/asset_manager
    core/classifier
    core/detector

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
