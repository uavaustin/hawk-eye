## Documentation
This file is meant to encompass some tips, tricks, and general documetation for `hawk_eye`.
By no means exhaustive yet, please adapt this file as time progresses.

### Setup
This repository works best with Linux, WSL, and Mac systems.
The `setup_linux.sh` script in this folder will let up Ubuntu and WSL systems.
To run the script, execute:
```./docs/setup_linux.sh``` 

inside the `hawk_eye` repository. This file also takes an optional argument to a python virtual environemnt:
```./docs/setup_linux.sh ~/path_to_venv```

Upon sucessful termination of this script, you need to then recieve access to
Google Cloud Storage. See you lead about gaining permissions, then run
`./docs/install_google_cloud.sh`
