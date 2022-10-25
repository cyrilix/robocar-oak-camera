"""
Camera event loop
"""
import abc
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

    def process(self, in_nn: dai.NNData, frame_ref) -> None:
        """
        Parse and publish result of NeuralNetwork result
        :param in_nn: NeuralNetwork result read from device
        :param frame_ref: Id of the frame where objects are been detected
        :return:
        """
        detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
        detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))
        # keep boxes bigger than threshold
        mask = detection_scores >= self._objects_threshold
        boxes = detection_boxes[mask]
        scores = detection_scores[mask]

        if boxes.shape[0] > 0:
            self._publish_objects(boxes, frame_ref, scores)

    def _publish_objects(self, boxes: np.array, frame_ref, scores: np.array) -> None:

        objects_msg = events.events_pb2.ObjectsMessage()
        objs = []
        for i in range(boxes.shape[0]):
            logger.debug("new object detected: %s", str(boxes[i]))
            objs.append(_bbox_to_object(boxes[i], scores[i].astype(float)))
        objects_msg.objects.extend(objs)
        objects_msg.frame_ref.name = frame_ref.name
        objects_msg.frame_ref.id = frame_ref.id
        objects_msg.frame_ref.created_at.FromDatetime(frame_ref.created_at.ToDatetime())
        logger.debug("publish object event to %s", self._objects_topic)
        self._mqtt_client.publish(topic=self._objects_topic,
                                  payload=objects_msg.SerializeToString(),
                                  qos=0,
                                  retain=False)


class FrameProcessError(Exception):
    """
    Error base for invalid frame processing

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        """
        :param message: explanation of the error
        """
        self.message = message


class FrameProcessor:
    """
    Processor for camera frames
    """

    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str):
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic

    def process(self, img: dai.ImgFrame) -> typing.Any:
        """
        Publish camera frames
        :param img:
        :return:
            id frame reference
        :raise:
            FrameProcessError if frame can't be processed
        """
        im_resize = img.getCvFrame()
        is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
        if not is_success:
            raise FrameProcessError("unable to process to encode frame to jpg")
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
        return frame_msg.id


class Source(abc.ABC):
    @abc.abstractmethod
    def get_stream_name(self) -> str:
        pass

    @abc.abstractmethod
    def link_preview(self, input_node: dai.Node.Input):
        pass


class ObjectDetectionNN:
    """
    Node to detect objects into image

    Read image as input and apply resize transformation before to run NN on it
    Result is available with 'get_stream_name()' stream
    """

    def __init__(self, pipeline: dai.Pipeline):
        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(_NN_PATH)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        self._detection_nn = detection_nn
        self._xout = self._configure_xout_nn(pipeline)
        self._detection_nn.out.link(self._xout.input)
        self._manip_image = self._configure_manip(pipeline)

    @staticmethod
    def _configure_manip(pipeline: dai.Pipeline) -> dai.node.ImageManip:
        # Resize image
        manip = pipeline.createImageManip()
        manip.initialConfig.setResize(_NN_WIDTH, _NN_HEIGHT)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        manip.initialConfig.setKeepAspectRatio(False)
        return manip

    @staticmethod
    def _configure_xout_nn(pipeline: dai.Pipeline) -> dai.node.XLinkOut:
        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        xout_nn.input.setBlocking(False)
        return xout_nn

    def get_stream_name(self) -> str:
        return self._xout.getStreamName()

    def get_input(self) -> dai.Node.Input:
        return self._manip_image.inputImage


class CameraSource(Source):
    """Image source based on camera preview"""

    def __init__(self, pipeline: dai.Pipeline, img_width: int, img_height: int):
        cam_rgb = pipeline.createColorCamera()
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        self._cam_rgb = cam_rgb
        self._xout_rgb = xout_rgb

        # Properties
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(width=img_width, height=img_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setFps(30)

        # link camera preview to output
        cam_rgb.preview.link(xout_rgb.input)

    def link_preview(self, input_node: dai.Node.Input):
        self._cam_rgb.preview.link(input_node)

    def get_stream_name(self) -> str:
        return self._xout_rgb.getStreamName()


class MqttSource(Source):
    """Image source based onto mqtt stream"""

    def __init__(self, device: dai.Device, pipeline: dai.Pipeline, mqtt_host: str, mqtt_topic: str,
                 mqtt_port: int = 1883, mqtt_qos: int = 0):
        self._mqtt_host = mqtt_host
        self._mqtt_port = mqtt_port

        self._client = mqtt.Client()
        self._client.user_data_set({"topic": mqtt_topic, "qos": str(mqtt_qos)})
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._img_in = pipeline.createXLinkIn()
        self._img_in.setStreamName("img_input")
        self._img_out = pipeline.createXLinkOut()
        self._img_out.setStreamName("img_output")
        self._img_in.out.link(self._img_out.input)

        self._img_in_queue = device.getInputQueue(self._img_in.getStreamName())

    def run(self):
        self._client.connect(host=self._mqtt_host, port=self._mqtt_port)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()

    @staticmethod
    def _on_connect(client: mqtt.Client, userdata: dict[str, str], flags, rc):
        # if we lose the connection and reconnect then subscriptions will be renewed.
        client.subscribe(topic=userdata["topic"], qos=int(userdata["qos"]))

    def _on_message(self, _: mqtt.Client, user_data: dict[str, str], msg: mqtt.MQTTMessage):
        frame_msg = events.events_pb2.FrameMessage()
        frame_msg.ParseFromString(msg.payload)

        frame = np.asarray(frame_msg.frame, dtype="uint8")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        nn_data = dai.NNData()
        nn_data.setLayer("data", _to_planar(frame, frame.shape()))
        self._img_in_queue.send(nn_data)

    def get_stream_name(self) -> str:
        return self._img_out.getStreamName()

    def link_preview(self, input_node: dai.Node.Input):
        self._img_in.out.link(input_node)


def _to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


class PipelineController:
    """
    Pipeline controller that drive camera device
    """

    def __init__(self, img_width: int, img_height: int, frame_processor: FrameProcessor,
                 object_processor: ObjectProcessor, camera: Source, object_node: ObjectDetectionNN):
        self._img_width = img_width
        self._img_height = img_height
        self._pipeline = self._configure_pipeline()
        self._frame_processor = frame_processor
        self._object_processor = object_processor
        self._camera = camera
        self._object_node = object_node
        self._stop = False

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()

        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        # Link preview to manip and manip to nn
        self._camera.link_preview(self._object_node.get_input())

        logger.info("pipeline configured")
        return pipeline

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
            q_rgb = device.getOutputQueue(name=self._camera.get_stream_name(), maxSize=queue_size, blocking=False)
            q_nn = device.getOutputQueue(name=self._object_node.get_stream_name(), maxSize=queue_size, blocking=False)

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
        try:
            frame_ref = self._frame_processor.process(in_rgb)
        except FrameProcessError as ex:
            logger.error("unable to process frame: %s", str(ex))
            return
        # Read NN result
        in_nn: dai.NNData = q_nn.get()
        self._object_processor.process(in_nn, frame_ref)

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
