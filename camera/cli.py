"""
Publish data from oak-lite device

Usage: rc-oak-camera [-u USERNAME | --mqtt-username=USERNAME] [--mqtt-password=PASSWORD] [--mqtt-broker=HOSTNAME] \
    [--mqtt-topic-robocar-oak-camera="TOPIC_CAMERA"] [--mqtt-topic-robocar-objects="TOPIC_OBJECTS"] \
    [--mqtt-client-id=CLIENT_ID] \
    [-H IMG_HEIGHT | --image-height=IMG_HEIGHT] [-W IMG_WIDTH | --image-width=IMG_width] \
    [-t OBJECTS_THRESHOLD | --objects-threshold=OBJECTS_THRESHOLD]

Options:
-h --help                                               Show this screen.
-u USERID --mqtt-username=USERNAME                      MQTT user
-p PASSWORD --mqtt-password=PASSWORD                    MQTT password
-b HOSTNAME --mqtt-broker=HOSTNAME                      MQTT broker host
-C CLIENT_ID --mqtt-client-id=CLIENT_ID                 MQTT client id
-c TOPIC_CAMERA --mqtt-topic-robocar-oak-camera=TOPIC_CAMERA        MQTT topic where to publish robocar-oak-camera frames
-o TOPIC_OBJECTS --mqtt-topic-robocar-objects=TOPIC_OBJECTS         MQTT topic where to publish objects detection results
-H IMG_HEIGHT --image-height=IMG_HEIGHT                 IMG_HEIGHT image height
-W IMG_WIDTH --image-width=IMG_width                    IMG_WIDTH image width
-t OBJECTS_THRESHOLD --objects-threshold=OBJECTS_THRESHOLD    OBJECTS_THRESHOLD threshold to filter objects detected
"""
import logging
import os
from . import depthai as cam
from docopt import docopt
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

default_client_id = "robocar-depthai"


def init_mqtt_client(broker_host: str, user: str, password: str, client_id: str) -> mqtt.Client:
    logger.info("Start part.py-robocar-oak-camera")
    client = mqtt.Client(client_id=client_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311)

    client.username_pw_set(user, password)
    logger.info("Connect to mqtt broker "+ broker_host)
    client.connect(host=broker_host, port=1883, keepalive=60)
    logger.info("Connected to mqtt broker")
    return client


def execute_from_command_line():
    logging.basicConfig(level=logging.INFO)

    args = docopt(__doc__)

    client = init_mqtt_client(broker_host=get_default_value(args["--mqtt-broker"], "MQTT_BROKER", "localhost"),
                              user=get_default_value(args["--mqtt-username"], "MQTT_USERNAME", ""),
                              password=get_default_value(args["--mqtt-password"], "MQTT_PASSWORD", ""),
                              client_id=get_default_value(args["--mqtt-client-id"], "MQTT_CLIENT_ID",
                                                          default_client_id),
                              )
    frame_topic = get_default_value(args["--mqtt-topic-robocar-oak-camera"], "MQTT_TOPIC_CAMERA", "/oak/camera_rgb")
    objects_topic = get_default_value(args["--mqtt-topic-robocar-objects"], "MQTT_TOPIC_OBJECTS", "/objects")

    frame_processor = cam.FramePublisher(mqtt_client=client,
                                         frame_topic=frame_topic,
                                         objects_topic=objects_topic,
                                         objects_threshold=float(get_default_value(args["--objects-threshold"],
                                                                                   "OBJECTS_THRESHOLD", 0.2)),
                                         img_width=int(get_default_value(args["--image-width"], "IMAGE_WIDTH", 160)),
                                         img_height=int(get_default_value(args["--image-height"], "IMAGE_HEIGHT", 120)))
    frame_processor.run()


def get_default_value(value, env_var: str, default_value) -> str:
    if value:
        return value
    if env_var in os.environ:
        return os.environ[env_var]
    return default_value
