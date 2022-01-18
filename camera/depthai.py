import logging
import paho.mqtt.client as mqtt

import events.events_pb2
from google.protobuf.timestamp_pb2 import Timestamp

import depthai as dai
import cv2

from threading import Thread

logger = logging.getLogger(__name__)


class FramePublisher(Thread):
    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str, img_width: int, img_height: int):
        super().__init__(name="FrameProcessor")
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic
        self._img_width = img_width
        self._img_height = img_height
        self._pipeline = self._configure_pipeline()

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Define sources and outputs
        manip = pipeline.create(dai.node.ImageManip)

        manip_out = pipeline.create(dai.node.XLinkOut)

        manip_out.setStreamName("manip")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        manip.initialConfig.setResize(self._img_width, self._img_height)

        # Linking
        cam_rgb.video.link(manip.inputImage)
        manip.out.link(manip_out.input)
        logger.info("pipeline configured")
        return pipeline

    def run(self):
        # Connect to device and start pipeline
        with dai.Device(self._pipeline) as device:
            # Queues
            queue_size = 8
            queue_manip = device.getOutputQueue("manip", queue_size)

            while True:
                try:
                    while queue_manip.has():
                        im_resize = queue_manip.get().getData().getCvFrame()

                        is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
                        byte_im = im_buf_arr.tobytes()

                        timestamp = Timestamp()
                        frame_msg = events.events_pb2.FrameMessage()
                        frame_msg.id = events.events_pb2.FrameRef()
                        frame_msg.id.name = "robocar-oak-camera-oak"
                        frame_msg.id.id = timestamp.ToMilliseconds()
                        frame_msg.id.created_at = timestamp.GetCurrentTime()
                        frame_msg.frame = byte_im

                        self._mqtt_client.publish(topic=self._frame_topic,
                                                  payload=frame_msg.SerializeToString(),
                                                  qos=0,
                                                  retain=False)

                except Exception as e:
                    logger.exception("unexpected error")
