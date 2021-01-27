#!/usr/bin/env bash

IS_MAC=$(uname -a)
if [[ $IS_MAC =~ "Darwin" ]]; then
    ARCHIVE=google-cloud-sdk-323.0.0-darwin-x86_64.tar.gz
else
    ARCHIVE=google-cloud-sdk-323.0.0-linux-x86_64.tar.gz
fi
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

sudo apt-get update && sudo apt-get install google-cloud-sdk
gcloud init
gcloud auth login --no-launch-browser
gcloud auth application-default login --no-launch-browser
