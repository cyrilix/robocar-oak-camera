"""
Camera event loop
"""
import datetime
import logging
import typing

import cv2
import depthai as dai
import numpy as np
import paho.mqtt.client as mqtt

import events.events_pb2

logger = logging.getLogger(__name__)

_NN_PATH = "/models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob"
_NN_WIDTH = 192
_NN_HEIGHT = 192


class ObjectProcessor:
    """
    Processor for Object detection
    """

    def __init__(self, mqtt_client: mqtt.Client, objects_topic: str, objects_threshold: float):
        self._mqtt_client = mqtt_client
        self._objects_topic = objects_topic
        self._objects_threshold = objects_threshold

    def process(self, in_nn: dai.NNData, frame_ref, frame_date: datetime.datetime) -> None:
        """
        Parse and publish result of NeuralNetwork result
        :param in_nn: NeuralNetwork result read from device
        :param frame_ref: Id of the frame where objects are been detected
        :param frame_date: Datetime of the frame used for detection
        :return:
        """
        detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
        detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))
        # keep boxes bigger than threshold
        mask = detection_scores >= self._objects_threshold
        boxes = detection_boxes[mask]
        scores = detection_scores[mask]

        if boxes.shape[0] > 0:
            self._publish_objects(boxes, frame_ref, frame_date, scores)

    def _publish_objects(self, boxes: np.array, frame_ref, now: datetime.datetime, scores: np.array) -> None:
        objects_msg = events.events_pb2.ObjectsMessage()
        objs = []
        for i in range(boxes.shape[0]):
            logger.debug("new object detected: %s", str(boxes[i]))
            objs.append(_bbox_to_object(boxes[i], scores[i].astype(float)))
        objects_msg.objects.extend(objs)
        objects_msg.frame_ref.name = frame_ref.name
        objects_msg.frame_ref.id = frame_ref.id
        objects_msg.frame_ref.created_at.FromDatetime(now)
        logger.debug("publish object event to %s", self._objects_topic)
        self._mqtt_client.publish(topic=self._objects_topic,
                                  payload=objects_msg.SerializeToString(),
                                  qos=0,
                                  retain=False)


class FrameProcessor:
    """
    Processor for camera frames
    """

    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str):
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic

    def process(self, img: dai.ImgFrame) -> (typing.Any, datetime.datetime):
        """
        Publish camera frames
        :param img:
        :return:
            id frame
            frame creation datetime
        """
        im_resize = img.getCvFrame()
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
        return frame_msg.id, now


class PipelineController:
    """
    Pipeline controller that drive camera device
    """

    def __init__(self, img_width: int, img_height: int, frame_processor: FrameProcessor,
                 object_processor: ObjectProcessor):
        self._img_width = img_width
        self._img_height = img_height
        self._pipeline = self._configure_pipeline()
        self._frame_processor = frame_processor
        self._object_processor = object_processor
        self._stop = False

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()

        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        detection_nn = self._configure_detection_nn(pipeline)
        xout_nn = self._configure_xout_nn(pipeline)

        # Resize image
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        manip.initialConfig.setKeepAspectRatio(False)

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(width=self._img_width, height=self._img_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)

        # Link preview to manip and manip to nn
        cam_rgb.preview.link(manip.inputImage)
        manip.out.link(detection_nn.input)

        # Linking to output
        cam_rgb.preview.link(xout_rgb.input)
        detection_nn.out.link(xout_nn.input)

        logger.info("pipeline configured")
        return pipeline

    @staticmethod
    def _configure_xout_nn(pipeline: dai.Pipeline) -> dai.node.XLinkOut:
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        xout_nn.input.setBlocking(False)
        return xout_nn

    @staticmethod
    def _configure_detection_nn(pipeline: dai.Pipeline) -> dai.node.NeuralNetwork:
        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(NN_PATH)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        return detection_nn

    def run(self) -> None:
        """
        Start event loop
        :return:
        """
        # Connect to device and start pipeline
        with dai.Device(self._pipeline) as device:
            logger.info('MxId: %s', device.getDeviceInfo().getMxId())
            logger.info('USB speed: %s', device.getUsbSpeed())
            logger.info('Connected cameras: %s', device.getConnectedCameras())
            logger.info("output queues found: %s", device.getOutputQueueNames())

            device.startPipeline()
            # Queues
            queue_size = 4
            q_rgb = device.getOutputQueue(name="rgb", maxSize=queue_size, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=queue_size, blocking=False)

            self._stop = False
            while True:
                if self._stop:
                    logger.info("stop loop event")
                    return
                try:
                    self._loop_on_camera_events(q_nn, q_rgb)
                # pylint: disable=broad-except # bad frame or event must not stop loop
                except Exception as ex:
                    logger.exception("unexpected error: %s", str(ex))

    def _loop_on_camera_events(self, q_nn: dai.DataOutputQueue, q_rgb: dai.DataOutputQueue):
        logger.debug("wait for new frame")

        # Wait for frame
        in_rgb: dai.ImgFrame = q_rgb.get()  # blocking call, will wait until a new data has arrived
        frame_msg, now = self._frame_processor.process(in_rgb)

        # Read NN result
        in_nn: dai.NNData = q_nn.get()
        self._object_processor.process(in_nn, frame_msg.id, now)

    def stop(self):
        """
        Stop event loop, if loop is not running, do nothing
        :return:
        """
        self._stop = True

def _bbox_to_object(bbox: np.array, score: float) -> events.events_pb2.Object:
    obj = events.events_pb2.Object()
    obj.type = events.events_pb2.TypeObject.ANY
    obj.top = bbox[0].astype(float)
    obj.right = bbox[3].astype(float)
    obj.bottom = bbox[2].astype(float)
    obj.left = bbox[1].astype(float)
    obj.confidence = score
    return obj
