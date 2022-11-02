FROM docker.io/library/python:3.10-slim as base

# Configure piwheels repo to use pre-compiled numpy wheels for arm
RUN echo -n "[global]\nextra-index-url=https://www.piwheels.org/simple\n" >> /etc/pip.conf

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

#################
FROM base as model-builder

RUN python3 -m pip install blobconverter

RUN mkdir -p /models

RUN blobconverter --zoo-name mobile_object_localizer_192x192 --zoo-type depthai --shaves 6 --version 2021.4 --output-dir /models || echo ""

#################
FROM base as builder

RUN apt-get install -y git && \
    pip3 install poetry==1.2.0 && \
    poetry self add "poetry-dynamic-versioning[plugin]"

ADD poetry.lock .
ADD pyproject.toml .
ADD camera camera
ADD README.md .

RUN poetry build

#################
FROM base

RUN mkdir /models
COPY --from=model-builder /models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob /models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob

COPY --from=builder dist/*.whl /tmp/
RUN pip3 install /tmp/*whl

WORKDIR /tmp
USER 1234

ENTRYPOINT ["/usr/local/bin/rc-oak-camera"]
