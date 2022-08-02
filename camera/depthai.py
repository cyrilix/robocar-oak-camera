import datetime
import logging
import paho.mqtt.client as mqtt

import events.events_pb2

import depthai as dai
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True
# Better handling for occlusions:
lr_check = True

class FramePublisher:
    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str, img_width: int, img_height: int):
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic
        self._img_width = img_width
        self._img_height = img_height
        self._depth = None
        self._pipeline = self._configure_pipeline()

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)

        xout_rgb.setStreamName("rgb")

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xout = pipeline.create(dai.node.XLinkOut)
        self._depth = depth

        xout.setStreamName("disparity")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(lr_check)
        depth.setExtendedDisparity(extended_disparity)
        depth.setSubpixel(subpixel)

        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = False
        config.postProcessing.spatialFilter.enable = False
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        #config.postProcessing.thresholdFilter.minRange = 400
        #config.postProcessing.thresholdFilter.maxRange = 15000
        config.postProcessing.decimationFilter.decimationFactor = 2
        depth.initialConfig.set(config)

        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.disparity.link(xout.input)

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

            # Output queue will be used to get the disparity frames from the outputs defined above
            q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

            while True:
                try:
                    logger.debug("wait for new frame")
                    inRgb = q_rgb.get()  # blocking call, will wait until a new data has arrived
                    inDisparity = q_disparity.get()
                    # im_resize = inRgb.getCvFrame()
                    im_resize = inDisparity.getCvFrame()

                    # Normalization for better visualization
                    im_resize = (im_resize * (255 / self._depth.initialConfig.getMaxDisparity())).astype(np.uint8)

                    # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
                    # im_resize = cv2.applyColorMap(im_resize, cv2.COLORMAP_JET)

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
