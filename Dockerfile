FROM docker.io/library/python:3.9-slim

# Configure piwheels repo to use pre-compiled numpy wheels for arm
RUN echo -n "[global]\nextra-index-url=https://www.piwheels.org/simple\n" >> /etc/pip.conf

RUN apt-get update && apt-get install -y libusb-1.0-0


RUN pip3 install numpy

ADD requirements.txt .

RUN pip3 install -r requirements.txt

ADD events .
ADD camera .
ADD setup.cfg .
ADD setup.py .


ENV PYTHON_EGG_CACHE=/tmp/cache
RUN python3 setup.py install && rm -rf /src

WORKDIR /tmp
USER 1234

ENTRYPOINT ["/usr/local/bin/rc-oak-camera"]
