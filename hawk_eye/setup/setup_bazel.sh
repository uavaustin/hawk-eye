# Get bazelisk
pushd $(mktemp -d)
if [[ $IS_MAC =~ "Darwin" ]]; then
    echo "Downloading Bazelisk for Darwin"
    curl -fL -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.2/bazelisk-darwin-amd64
else
    curl -fL -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.7.1/bazelisk-linux-amd64
fi

chmod +x bazel
mv bazel /usr/local/bin
popd
