#!/usr/bin/env bash

IS_MAC=$(uname -a)
if [[ $IS_MAC =~ "Darwin" ]]; then
    ARCHIVE=google-cloud-sdk-323.0.0-darwin-x86_64.tar.gz
else
    ARCHIVE=google-cloud-sdk-323.0.0-linux-x86_64.tar.gz
fi
pushd $(mktemp -d)
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/$ARCHIVE
tar -xf $ARCHIVE
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init --skip-diagnostics --console-only
./google-cloud-sdk/bin/gcloud components update
./google-cloud-sdk/bin/gcloud auth login
popd
