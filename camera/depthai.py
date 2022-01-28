import datetime
import logging
import paho.mqtt.client as mqtt

import events.events_pb2

import depthai as dai
import cv2

logger = logging.getLogger(__name__)


class FramePublisher:
    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str, img_width: int, img_height: int):
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic
        self._img_width = img_width
        self._img_height = img_height
        self._pipeline = self._configure_pipeline()

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)

        xout_rgb.setStreamName("rgb")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(width=self._img_width, height=self._img_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)

        # Linking
        cam_rgb.preview.link(xout_rgb.input)
        logger.info("pipeline configured")
        return pipeline

    def run(self):
        # Connect to device and start pipeline
        with dai.Device(self._pipeline) as device:
            logger.info('MxId: %s', device.getDeviceInfo().getMxId())
            logger.info('USB speed: %s', device.getUsbSpeed())
            logger.info('Connected cameras: %s', device.getConnectedCameras())

            logger.info("output queues found: %s", device.getOutputQueueNames())

            device.startPipeline()
            # Queues
            queue_size = 4
            q_rgb = device.getOutputQueue("rgb", maxSize=queue_size, blocking=False)

            while True:
                try:
                    logger.debug("wait for new frame")
                    inRgb = q_rgb.get()  # blocking call, will wait until a new data has arrived

                    im_resize = inRgb.getCvFrame()

                    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
                    byte_im = im_buf_arr.tobytes()

                    now = datetime.datetime.now()
                    frame_msg = events.events_pb2.FrameMessage()
                    frame_msg.id.name = "robocar-oak-camera-oak"
                    frame_msg.id.id = str(int(now.timestamp() * 1000))
                    frame_msg.id.created_at.FromDatetime(now)
                    frame_msg.frame = byte_im

                    logger.debug("publish frame event to %s", self._frame_topic)
                    self._mqtt_client.publish(topic=self._frame_topic,
                                              payload=frame_msg.SerializeToString(),
                                              qos=0,
                                              retain=False)

                except Exception as e:
                    logger.exception("unexpected error: %s", str(e))
