docker build -t hawk-eye --build-arg USER=$USER -f hawk_eye/setup/Dockerfile .

docker run --rm -ti -v $PWD:/build -v /home:/home -u $(id -u):$(id -g) hawk-eye:latest /bin/bash
