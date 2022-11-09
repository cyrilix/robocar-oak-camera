"""
Mqtt gateway for oak-lite device
"""
import argparse
import logging
import os
import signal
import typing, types

import depthai as dai
import paho.mqtt.client as mqtt

from . import depthai as cam  # pylint: disable=reimported

logger = logging.getLogger(__name__)

_DEFAULT_CLIENT_ID = "robocar-depthai"


def _parse_args_cli() -> argparse.Namespace:
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
                        default=_get_env_value("MQTT_CLIENT_ID", _DEFAULT_CLIENT_ID))
    parser.add_argument("-c", "--mqtt-topic-robocar-oak-camera",
                        help="MQTT topic where to publish robocar-oak-camera frames",
                        default=_get_env_value("MQTT_TOPIC_CAMERA", "/oak/camera_rgb"))
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
    parser.add_argument("--log", help="Log level",
                        type=str,
                        default="info",
                        choices=["info", "debug"])
    args = parser.parse_args()
    return args


def _init_mqtt_client(broker_host: str, broker_port: int, user: str, password: str, client_id: str) -> mqtt.Client:
    logger.info("Start part.py-robocar-oak-camera")
    client = mqtt.Client(client_id=client_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311)

    client.username_pw_set(user, password)
    logger.info("Connect to mqtt broker %s", broker_host)
    client.connect(host=broker_host, port=broker_port, keepalive=60)
    logger.info("Connected to mqtt broker")
    return client


def execute_from_command_line() -> None:
    """
    Cli entrypoint
    :return:
    """

    args = _parse_args_cli()
    if args.log == "info":
        logging.basicConfig(level=logging.INFO)
    elif args.log == "debug":
        logging.basicConfig(level=logging.DEBUG)

    client = _init_mqtt_client(broker_host=args.mqtt_broker_host,
                               broker_port=args.mqtt_broker_port,
                               user=args.mqtt_username,
                               password=args.mqtt_password,
                               client_id=args.mqtt_client_id,
                               )
    frame_processor = cam.FrameProcessor(mqtt_client=client, frame_topic=args.mqtt_topic_robocar_oak_camera)
    object_processor = cam.ObjectProcessor(mqtt_client=client,
                                           objects_topic=args.mqtt_topic_robocar_objects,
                                           objects_threshold=args.objects_threshold)

    pipeline = dai.Pipeline()
    pipeline_controller = cam.PipelineController(pipeline=pipeline,
                                                 frame_processor=frame_processor,
                                                 object_processor=object_processor,
                                                 object_node=cam.ObjectDetectionNN(pipeline=pipeline),
                                                 camera=cam.CameraSource(pipeline=pipeline,
                                                                         img_width=args.image_width,
                                                                         img_height=args.image_width,
                                                                         ))

    def sigterm_handler(signum: int, frame: typing.Optional[
        types.FrameType]) -> None:  # pylint: disable=unused-argument  # need to implement handler signature
        logger.info("exit on SIGTERM")
        pipeline_controller.stop()

    signal.signal(signal.SIGTERM, sigterm_handler)
    pipeline_controller.run()


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
