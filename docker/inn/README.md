# Innovatrics Maskrcnn-Benchmark

## Training

Follow the main [README.md](../../README.md).
Supported data is actually COCO only.
The default backbone is based on `configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`.
Every specific training should have its own configuration in [configs](../../configs) folder

## Inference

Start `lab` in the docker container environment and copy the generated url including the token into your browser.

### Start the lab service

~~~bash
cd docker/inn
docker-compose run --rm --service-ports lab
~~~

In lab go to `demo` folder and launch the specific `*-inference.ipynb` notebook.

## Development Environment

All dependencies are provided in docker image.
Follow instructions to build docker image and running a container based on such image.

### Build maskrcnn-benchmark Image

Docker build is based on [BuildKit](https://docs.docker.com/develop/develop-images/build_enhancements/) due to the possibility to access private git repositories during building the image.
Two images are available.

- **dev-maskrcnn-benchmark** designated for development (maskrcnn-benchmark python codes are editable and all changes are permanent).
- **app-maskrcnn-benchmark** with backed maskrcnn-benchmark almost ready for *deployment*

#### dev-maskrcnn-benchmark

Dev image allows to work with maskrcnn-benchmark codes without the need to rebuild the image with every new change.
To build the `dev-maskrcnn-benchmark` image run

~~~bash
DOCKER_BUILDKIT=1 docker build \
    --target dev-image \
    --ssh default \
    --add-host=git.ba.innovatrics.net:$(getent hosts git.ba.innovatrics.net | cut -d' ' -f1) \
    -t dev-maskrcnn-benchmark:1.0 \
    -f docker/inn/Dockerfile .
~~~

which forwards your ssh agent during the build process and allows to clone several private git Innovatrics repositories.

#### app-maskrcnn-benchmark

App image which will be further reworked for deployment.

~~~bash
DOCKER_BUILDKIT=1 docker build \
    --target app-image \
    --ssh default \
    --add-host=git.ba.innovatrics.net:$(getent hosts git.ba.innovatrics.net | cut -d' ' -f1) \
    -t app-maskrcnn-benchmark:1.0 \
    -f docker/inn/Dockerfile .
~~~

#### Python Dependencies

Skip this section in a case no new dependencies need to be changed or added.

Aside is a docker image for freezing the python dependencies based on Python 3.6.8.
[pip-tools](https://pypi.org/project/pip-tools/) is used to freeze all the dependencies based on the
[requirements.inn](requirements.inn) file which are stored as [requirements.txt](requirements.txt) and later used for building maskrcnn-benchmark environment.

Build the image with

~~~bash
DOCKER_BUILDKIT=1 docker build \
    --ssh default \
    --add-host=git.ba.innovatrics.net:$(getent hosts git.ba.innovatrics.net | cut -d' ' -f1) \
    -t deps-maskrcnn-benchmark:1.0 \
    -f docker/inn/deps/Dockerfile .
~~~

Get the requirements.txt

~~~bash
docker run --rm -it -v /path/to/dest:/mnt deps-maskrcnn-benchmark bash
cp /opt/requirements.txt /mnt/requirements.txt
~~~
