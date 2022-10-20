import logging
import os
from . import depthai as cam
import paho.mqtt.client as mqtt
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

default_client_id = "robocar-depthai"


def init_mqtt_client(broker_host: str, broker_port, user: str, password: str, client_id: str) -> mqtt.Client:
    logger.info("Start part.py-robocar-oak-camera")
    client = mqtt.Client(client_id=client_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311)

    client.username_pw_set(user, password)
    logger.info("Connect to mqtt broker "+ broker_host)
    client.connect(host=broker_host, port=broker_port, keepalive=60)
    logger.info("Connected to mqtt broker")
    return client


def execute_from_command_line():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--mqtt-username",
                        help="MQTT user",
                        default=_get_env_value("MQTT_USERNAME", ""))
    parser.add_argument("-p", "--mqtt-password",
                        help="MQTT password",
                        default=_get_env_value("MQTT_PASSWORD", ""))
    parser.add_argument("-b", "--mqtt-broker-host",
                        help="MQTT broker host",
                        default=_get_env_value("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument("-P", "--mqtt-broker-port",
                        help="MQTT broker port",
                        type=int,
                        default=_get_env_int_value("MQTT_BROKER_PORT", 1883))
    parser.add_argument("-C", "--mqtt-client-id",
                        help="MQTT client id",
                        default=_get_env_value("MQTT_CLIENT_ID", default_client_id))
    parser.add_argument("-c", "--mqtt-topic-robocar-oak-camera",
                        help="MQTT topic where to publish robocar-oak-camera frames",
                        default=_get_env_value("MQTT_TOPIC_CAMERA","/oak/camera_rgb"))
    parser.add_argument("-o", "---mqtt-topic-robocar-objects",
                        help="MQTT topic where to publish objects detection results",
                        default=_get_env_value("MQTT_TOPIC_OBJECTS", "/objects"))
    parser.add_argument("-t", "--objects-threshold",
                        help="threshold to filter detected objects",
                        type=float,
                        default=_get_env_float_value("OBJECTS_THRESHOLD", 0.2))
    parser.add_argument("-H", "--image-height", help="image height",
                        type=int,
                        default=_get_env_int_value("IMAGE_HEIGHT", 120))
    parser.add_argument("-W", "--image-width", help="image width",
                        type=int,
                        default=_get_env_int_value("IMAGE_WIDTH", 126))

    args = parser.parse_args()

    client = init_mqtt_client(broker_host=args.mqtt_broker_host,
                              broker_port=args.mqtt_broker_port,
                              user=args.mqtt_username,
                              password=args.mqtt_password,
                              client_id=args.mqtt_client_id,
                              )
    frame_processor = cam.FramePublisher(mqtt_client=client,
                                         frame_topic=args.mqtt_topic_robocar_oak_camera,
                                         objects_topic=args.mqtt_topic_robocar_objects,
                                         objects_threshold=args.objects_threshold,
                                         img_width=args.image_width,
                                         img_height=args.image_height)
    frame_processor.run()


def _get_env_value(env_var: str, default_value: str) -> str:
    if env_var in os.environ:
        return os.environ[env_var]
    return default_value


def _get_env_int_value(env_var: str, default_value: int) -> int:
    value = _get_env_value(env_var, str(default_value))
    return int(value)


def _get_env_float_value(env_var: str, default_value: float) -> float:
    value = _get_env_value(env_var, str(default_value))
    return float(value)
