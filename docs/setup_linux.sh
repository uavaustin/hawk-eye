#!/usr/bin/env bash

sudo apt update && sudo apt install python3

pushd $(mktemp -d)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py
popd
python3 -m pip install --upgrade pip

# Create a virtual environment for hawk_eye
python3 -m venv ~/python_envs/hawk_eye
source ~/python_envs/hawk_eye/bin/activate

python3 -m pip install -U Cython==0.29.21 numpy==1.19.1
python3 -m pip install -U -r requirements.txt

GPU=$(sudo lspci | grep --quiet ' VGA ' | cut -d" " -f 1 | xargs -i lspci -v -s {})
echo $GPU
if [[ $GPU ]];
then
    echo "GPU found."
    python3 -m pip install -U -r requirements-gpu.txt
    pushd $(mktemp -d)
    rm -rf apex
    git clone --recursive https://github.com/NVIDIA/apex
    pushd apex
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
    git checkout 1ff54b8fed441c39dac181091b44fecdca31a403
    CUDA_HOME="/usr/local/cuda-10.1" pip install -v --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
    popd
else
    echo "No GPU found."
    python3 -m pip install -U -r requirements-cpu.txt
fi

# Get bazelisk
pushd $(mktemp -d)
wget -O bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.1/bazelisk-linux-amd64
chmod 755 bazel
sudo mv bazel /usr/local/bin
popd


# Setup gsutil. Taken from https://cloud.google.com/storage/docs/gsutil_install#deb

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
gcloud init
