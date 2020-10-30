# Copyright 2017 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# Template script to download GCS file and validate its digest
set -e

echo "Downloading %{BUCKET}/%{FILE} to %{DOWNLOAD_PATH}"
gsutil cp %{BUCKET}/%{FILE} %{DOWNLOAD_PATH}

digest=$(sha256sum %{DOWNLOAD_PATH} | head -c 64)
if [ $digest != %{SHA256} ]; then
  echo "actual digest: $digest, expected: %{SHA256}"
  exit -1
else
  exit 0
fi
