FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /build

ARG USER

COPY hawk_eye/setup/setup_env.sh hawk_eye/setup/
COPY hawk_eye/setup/setup_bazel.sh hawk_eye/setup/
COPY hawk_eye/setup/requirements.txt hawk_eye/setup/
COPY hawk_eye/setup/requirements-cpu.txt hawk_eye/setup/

RUN hawk_eye/setup/setup_env.sh
RUN bash hawk_eye/setup/setup_bazel.sh

RUN echo $USER
RUN useradd -ms /bin/bash $USER
WORKDIR /home/$USER
