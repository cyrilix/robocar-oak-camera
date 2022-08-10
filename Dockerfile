FROM docker.io/library/python:3.9-slim AS model

RUN python3 -m pip install blobconverter

RUN mkdir -p /models

RUN blobconverter --zoo-name mobile_object_localizer_192x192 --zoo-type depthai --shaves 6 --version 2021.4 --output-dir /models || echo ""
RUN ls /models
#######
FROM docker.io/library/python:3.9-slim

# Configure piwheels repo to use pre-compiled numpy wheels for arm
RUN echo -n "[global]\nextra-index-url=https://www.piwheels.org/simple\n" >> /etc/pip.conf

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip3 install numpy

RUN mkdir /models

COPY --from=model /models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob /models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob
ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ADD events events
ADD camera camera
ADD setup.cfg setup.cfg
ADD setup.py setup.py

ENV PYTHON_EGG_CACHE=/tmp/cache
RUN python3 setup.py install

WORKDIR /tmp
USER 1234

ENTRYPOINT ["/usr/local/bin/rc-oak-camera"]
