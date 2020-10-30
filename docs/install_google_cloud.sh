#!/usr/bin/env bash

IS_MAC=$(uname -a)
if [[ $IS_MAC =~ "Darwin" ]]; then
    curl https://sdk.cloud.google.com | bash
    echo "Exiting the shell. Please run 'gcloud init'"
    exec -l $SHELL
else
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    sudo apt-get install -y apt-transport-https ca-certificates gnupg
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update -y && sudo apt-get install -y google-cloud-sdk
fi
gcloud init
gcloud auth application-default login

export GOOGLE_APPLICATION_CREDENTIALS="~/zeta-time-285220-29e924f9a463.json"
