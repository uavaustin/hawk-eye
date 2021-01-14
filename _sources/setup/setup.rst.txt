This projects works best with Linux, WSL, and Mac systems. The `setup_linux.sh` script
in this folder will set up Ubuntu and WSL systems. To run the script, execute::

    ./docs/setup_linux.sh


inside of the `hawk_eye` repository. This file also takes an optional argument to a python
virtual environemnt::

    ./docs/setup_linux.sh ~/path_to_venv


Upon sucessful termination of this script, you need to then recieve access to Google
Cloud Storage. See your lead about gaining permissions, then run::

    ./docs/install_google_cloud.sh

To ensure you are all setup and ready to code, you can test code by running::

    bazel test //...
