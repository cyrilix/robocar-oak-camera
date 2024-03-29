"""
Camera event loop
"""
import abc
import datetime
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import List, Any

import cv2
import depthai as dai
import events.events_pb2 as evt
import numpy as np
import numpy.typing as npt
import paho.mqtt.client as mqtt
from depthai import Device

logger = logging.getLogger(__name__)

_NN_PATH = "/models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob"
_NN_WIDTH = 192
_NN_HEIGHT = 192

_PREVIEW_WIDTH = 640
_PREVIEW_HEIGHT = 480

_CAMERA_BASELINE_IN_MM = 75


class ObjectProcessor:
    """
    Processor for Object detection
    """

    def __init__(self, mqtt_client: mqtt.Client, objects_topic: str, objects_threshold: float):
        self._mqtt_client = mqtt_client
        self._objects_topic = objects_topic
        self._objects_threshold = objects_threshold

    def process(self, in_nn: dai.NNData, frame_ref: evt.FrameRef) -> None:
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

    def _publish_objects(self, boxes: npt.NDArray[np.float64], frame_ref: evt.FrameRef, scores: npt.NDArray[np.float64]) -> None:
        objects_msg = evt.ObjectsMessage()
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

    def process(self, img: dai.ImgFrame) -> Any:
        """
        Publish camera frames
        :param img: image read from camera
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
        frame_msg = evt.FrameMessage()
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


class DisparityProcessor:
    """
       Processor for camera frames
       """

    def __init__(self, mqtt_client: mqtt.Client, disparity_topic: str):
        self._mqtt_client = mqtt_client
        self._disparity_topic = disparity_topic

    def process(self, img: dai.ImgFrame, frame_ref: evt.FrameRef, focal_length_in_pixels: float,
                baseline_mm: float = _CAMERA_BASELINE_IN_MM) -> None:
        im_frame = img.getCvFrame()
        is_success, im_buf_arr = cv2.imencode(".jpg", im_frame)
        if not is_success:
            raise FrameProcessError("unable to process to encode frame to jpg")
        byte_im = im_buf_arr.tobytes()

        disparity_msg = evt.DisparityMessage()
        disparity_msg.disparity = byte_im
        disparity_msg.frame_ref.name = frame_ref.name
        disparity_msg.frame_ref.id = frame_ref.id
        disparity_msg.frame_ref.created_at.FromDatetime(frame_ref.created_at.ToDatetime())
        disparity_msg.focal_length_in_pixels = focal_length_in_pixels
        disparity_msg.baseline_in_mm = baseline_mm

        self._mqtt_client.publish(topic=self._disparity_topic,
                                  payload=disparity_msg.SerializeToString(),
                                  qos=0,
                                  retain=False)


class Source(abc.ABC):
    """Base class for image source"""

    @abc.abstractmethod
    def get_stream_name(self) -> str:
        """
        Queue/stream name to use to get data

        :return: steam name
        """

    @abc.abstractmethod
    def link(self, input_node: dai.Node.Input) -> None:
        """
        Link this source to the input node

        :param: input_node:  input node to link
        """


class ObjectDetectionNN:
    """
    Node to detect objects into image

    Read image as input and apply resize transformation before to run NN on it
    Result is available with 'get_stream_name()' stream
    """

    def __init__(self, pipeline: dai.Pipeline):
        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(pathlib.Path(_NN_PATH))
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        self._detection_nn = detection_nn
        self._xout = self._configure_xout_nn(pipeline)
        self._detection_nn.out.link(self._xout.input)
        self._manip_image = self._configure_manip(pipeline)
        self._manip_image.out.link(self._detection_nn.input)

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
        """
        Queue/stream name to use to get data

        :return: stream name
        """
        return self._xout.getStreamName()

    def get_input(self) -> dai.Node.Input:
        """
        Get input node to use to link with source node
        :return: input to link with source output, see Source.link()
        """
        return self._manip_image.inputImage


class CameraSource(Source):
    """Image source based on camera preview"""

    def __init__(self, pipeline: dai.Pipeline, img_width: int, img_height: int, fps: int):
        self._cam_rgb = pipeline.createColorCamera()
        self._xout_rgb = pipeline.createXLinkOut()
        self._xout_rgb.setStreamName("rgb")

        # Properties
        self._cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self._cam_rgb.setPreviewSize(width=_PREVIEW_WIDTH, height=_PREVIEW_HEIGHT)
        self._cam_rgb.setInterleaved(False)
        self._cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self._cam_rgb.setFps(fps)
        self._resize_manip = self._configure_manip(pipeline=pipeline, img_width=img_width, img_height=img_height)

        # link camera preview to output
        self._cam_rgb.preview.link(self._resize_manip.inputImage)
        self._resize_manip.out.link(self._xout_rgb.input)

    def link(self, input_node: dai.Node.Input) -> None:
        self._cam_rgb.preview.link(input_node)

    def get_stream_name(self) -> str:
        return self._xout_rgb.getStreamName()

    @staticmethod
    def _configure_manip(pipeline: dai.Pipeline, img_width: int, img_height: int) -> dai.node.ImageManip:
        # Resize image
        manip = pipeline.createImageManip()
        manip.initialConfig.setResize(img_width, img_height)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        manip.initialConfig.setKeepAspectRatio(False)
        return manip


class StereoDepthPostFilter(abc.ABC):
    @abc.abstractmethod
    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        pass


class MedianFilter(StereoDepthPostFilter):
    """
    This is a non-edge preserving Median filter, which can be used to reduce noise and smoothen the depth map.
    Median filter is implemented in hardware, so it’s the fastest filter.
    """
    def __init__(self, value: dai.MedianFilter = dai.MedianFilter.KERNEL_7x7) -> None:
        self._value = value

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.median.value = self._value


class SpeckleFilter(StereoDepthPostFilter):
    """
    Speckle Filter is used to reduce the speckle noise. Speckle noise is a region with huge variance between
    neighboring disparity/depth pixels, and speckle filter tries to filter this region.
    """
    def __init__(self, enable: bool = True, speckle_range: int = 50) -> None:
        """
        :param enable: Whether to enable or disable the filter.
        :param speckle_range: Speckle search range.
        """
        self._enable = enable
        self._speckle_range = speckle_range

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.speckleFilter.enable = self._enable
        config.postProcessing.speckleFilter.speckleRange = self._speckle_range


class TemporalFilter(StereoDepthPostFilter):
    """
    Temporal Filter is intended to improve the depth data persistency by manipulating per-pixel values based on
    previous frames. The filter performs a single pass on the data, adjusting the depth values while also updating the
    tracking history. In cases where the pixel data is missing or invalid, the filter uses a user-defined persistency
    mode to decide whether the missing value should be rectified with stored data. Note that due to its reliance on
    historic data the filter may introduce visible blurring/smearing artifacts, and therefore is best-suited for
    static scenes.
    """
    def __init__(self,
                 enable: bool = True,
                 persistencyMode: dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4,
                 alpha: float = 0.4,
                 delta: int = 0):
        """
        :param enable: Whether to enable or disable the filter.
        :param persistencyMode: Persistency mode. If the current disparity/depth value is invalid, it will be replaced
        by an older value, based on persistency mode.
        :param alpha: The Alpha factor in an exponential moving average with Alpha=1 - no filter.
        Alpha = 0 - infinite filter. Determines the extent of the temporal history that should be averaged.
        :param delta: Step-size boundary. Establishes the threshold used to preserve surfaces (edges).
        If the disparity value between neighboring pixels exceed the disparity threshold set by this delta parameter,
        then filtering will be temporarily disabled. Default value 0 means auto: 3 disparity integer levels.
        In case of subpixel mode it’s 3*number of subpixel levels.
        """
        self._enable = enable
        self._persistencyMode = persistencyMode
        self._alpha = alpha
        self._delta = delta

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.temporalFilter.enable = self._enable
        config.postProcessing.temporalFilter.persistencyMode = self._persistencyMode
        config.postProcessing.temporalFilter.alpha = self._alpha
        config.postProcessing.temporalFilter.delta = self._delta


class SpatialFilter(StereoDepthPostFilter):
    """
    Spatial Edge-Preserving Filter will fill invalid depth pixels with valid neighboring depth pixels. It performs a
    series of 1D horizontal and vertical passes or iterations, to enhance the smoothness of the reconstructed data.
    """
    def __init__(self,
                 enable: bool = True,
                 hole_filling_radius: int = 2,
                 alpha: float = 0.5,
                 delta: int = 0,
                 num_iterations: int = 1):
        """
        :param enable: Whether to enable or disable the filter.
        :param hole_filling_radius: An in-place heuristic symmetric hole-filling mode applied horizontally during
        the filter passes. Intended to rectify minor artefacts with minimal performance impact. Search radius for
        hole filling.
        :param alpha: The Alpha factor in an exponential moving average with Alpha=1 - no filter.
        Alpha = 0 - infinite filter. Determines the amount of smoothing.
        :param delta: Step-size boundary. Establishes the threshold used to preserve “edges”. If the disparity value
        between neighboring pixels exceed the disparity threshold set by this delta parameter, then filtering will be
        temporarily disabled. Default value 0 means auto: 3 disparity integer levels. In case of subpixel mode it’s
        3*number of subpixel levels.
        :param num_iterations: Number of iterations over the image in both horizontal and vertical direction.
        """
        self._enable = enable
        self._hole_filling_radius = hole_filling_radius
        self._alpha = alpha
        self._delta = delta
        self._num_iterations = num_iterations

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.spatialFilter.enable = self._enable
        config.postProcessing.spatialFilter.holeFillingRadius = self._hole_filling_radius
        config.postProcessing.spatialFilter.alpha = self._alpha
        config.postProcessing.spatialFilter.delta = self._delta
        config.postProcessing.spatialFilter.numIterations = self._num_iterations


class ThresholdFilter(StereoDepthPostFilter):
    """
    Threshold Filter filters out all disparity/depth pixels outside the configured min/max threshold values.
    """
    def __init__(self, min_range: int = 400, max_range: int = 15000):
        """
        :param min_range: Minimum range in depth units. Depth values under this value are invalidated.
        :param max_range: Maximum range in depth units. Depth values over this value are invalidated.
        """
        self._min_range = min_range
        self._max_range = max_range

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.thresholdFilter.minRange = self._min_range
        config.postProcessing.thresholdFilter.maxRange = self._max_range


class DecimationFilter(StereoDepthPostFilter):
    """
    Decimation Filter will sub-samples the depth map, which means it reduces the depth scene complexity and allows
    other filters to run faster. Setting decimationFactor to 2 will downscale 1280x800 depth map to 640x400.
    """
    def __init__(self,
                 decimation_factor: int = 1,
                 decimation_mode: dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING
                 ):
        """
        :param decimation_factor: Decimation factor. Valid values are 1,2,3,4. Disparity/depth map x/y resolution will
        be decimated with this value.
        :param decimation_mode: Decimation algorithm type.
        """
        self._decimation_factor = decimation_factor
        self._mode = decimation_mode

    def apply(self, config: dai.RawStereoDepthConfig) -> None:
        config.postProcessing.decimationFilter.decimationFactor = self._decimation_factor
        config.postProcessing.decimationFilter.decimationMode = self._mode


class DepthSource(Source):
    def __init__(self, pipeline: dai.Pipeline,
                 extended_disparity: bool = False,
                 subpixel: bool = False,
                 lr_check: bool = True,
                 stereo_filters: List[StereoDepthPostFilter] = []
                 ) -> None:
        """
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True
        """
        self._monoLeft = pipeline.create(dai.node.MonoCamera)
        self._monoRight = pipeline.create(dai.node.MonoCamera)
        self._depth = pipeline.create(dai.node.StereoDepth)
        self._xout_disparity = pipeline.create(dai.node.XLinkOut)

        self._xout_disparity.setStreamName("disparity")

        # Properties
        self._monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self._monoLeft.setCamera("left")
        self._monoLeft.out.link(self._depth.left)
        self._monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self._monoRight.setCamera("right")
        self._monoRight.out.link(self._depth.right)

        # Create a node that will produce the depth map
        # (using disparity output as it's easier to visualize depth this way)
        self._depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        self._depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self._depth.setLeftRightCheck(lr_check)
        self._depth.setExtendedDisparity(extended_disparity)
        self._depth.setSubpixel(subpixel)
        self._depth.disparity.link(self._xout_disparity.input)

        if len(stereo_filters) > 0:
            # Configure post-processing filters
            config = self._depth.initialConfig.get()
            for filter in stereo_filters:
                filter.apply(config)
            self._depth.initialConfig.set(config)

    def get_stream_name(self) -> str:
        return self._xout_disparity.getStreamName()

    def link(self, input_node: dai.Node.Input) -> None:
        self._depth.disparity.link(input_node)


@dataclass
class MqttConfig:
    """MQTT configuration"""
    host: str
    topic: str
    port: int = 1883
    qos: int = 0


class MqttSource(Source):
    """Image source based onto mqtt stream"""

    def __init__(self, device: Device, pipeline: dai.Pipeline, mqtt_config: MqttConfig):
        self._mqtt_config = mqtt_config
        self._client = mqtt.Client()
        self._client.user_data_set(mqtt_config)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._img_in = pipeline.createXLinkIn()
        self._img_in.setStreamName("img_input")
        self._img_out = pipeline.createXLinkOut()
        self._img_out.setStreamName("img_output")
        self._img_in.out.link(self._img_out.input)

        self._img_in_queue = device.getInputQueue(self._img_in.getStreamName())

    def run(self) -> None:
        """ Connect and start mqtt loop """
        self._client.connect(host=self._mqtt_config.host, port=self._mqtt_config.port)
        self._client.loop_start()

    def stop(self) -> None:
        """Stop and disconnect mqtt loop"""
        self._client.loop_stop()
        self._client.disconnect()

    @staticmethod
    # pylint: disable=unused-argument
    def _on_connect(client: mqtt.Client, userdata: MqttConfig, flags: Any,
                    result_connection: Any) -> None:
        # if we lose the connection and reconnect then subscriptions will be renewed.
        client.subscribe(topic=userdata.topic, qos=userdata.qos)

    # pylint: disable=unused-argument
    def _on_message(self, _: mqtt.Client, user_data: MqttConfig, msg: mqtt.MQTTMessage) -> None:
        frame_msg = evt.FrameMessage()
        frame_msg.ParseFromString(msg.payload)

        frame = np.asarray(frame_msg.frame, dtype="uint8")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        nn_data = dai.NNData()
        nn_data.setLayer("data", _to_planar(frame, (300, 300)))
        self._img_in_queue.send(nn_data)

    def get_stream_name(self) -> str:
        return self._img_out.getStreamName()

    def link(self, input_node: dai.Node.Input) -> None:
        self._img_in.out.link(input_node)


def _to_planar(arr: npt.NDArray[np.uint8], shape: tuple[int, int]) -> list[int]:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


class PipelineController:
    """
    Pipeline controller that drive camera device
    """

    def __init__(self, frame_processor: FrameProcessor,
                 object_processor: ObjectProcessor, disparity_processor: DisparityProcessor,
                 camera: Source, depth_source: Source, object_node: ObjectDetectionNN,
                 pipeline: dai.Pipeline):
        self._frame_processor = frame_processor
        self._object_processor = object_processor
        self._disparity_processor = disparity_processor
        self._camera = camera
        self._depth_source = depth_source
        self._object_node = object_node
        self._stop = False
        self._pipeline = pipeline
        self._configure_pipeline()
        self._focal_length_in_pixels: float | None = None

    def _configure_pipeline(self) -> None:
        logger.info("configure pipeline")

        self._pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        # Link preview to manip and manip to nn
        self._camera.link(self._object_node.get_input())

        logger.info("pipeline configured")

    def run(self) -> None:
        """
        Start event loop
        :return:
        """
        # Connect to device and start pipeline
        with Device(pipeline=self._pipeline) as dev:
            logger.info('MxId: %s', dev.getDeviceInfo().getMxId())
            logger.info('USB speed: %s', dev.getUsbSpeed())
            logger.info('Connected cameras: %s', str(dev.getConnectedCameras()))
            logger.info("output queues found: %s", str(''.join(dev.getOutputQueueNames())))  # type: ignore

            calib_data = dev.readCalibration()
            intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
            self._focal_length_in_pixels = intrinsics[0][0]
            logger.info('Right mono camera focal length in pixels: %s', self._focal_length_in_pixels)

            dev.startPipeline()
            # Queues
            queue_size = 4
            q_rgb = dev.getOutputQueue(name=self._camera.get_stream_name(), maxSize=queue_size,  # type: ignore
                                       blocking=False)
            q_nn = dev.getOutputQueue(name=self._object_node.get_stream_name(), maxSize=queue_size,  # type: ignore
                                      blocking=False)
            if self._disparity_processor is not None:
                q_disparity = dev.getOutputQueue(name=self._depth_source.get_stream_name(), maxSize=queue_size,  # type: ignore
                                                 blocking=False)
            else:
                q_disparity = None

            start_time = time.time()
            counter = 0
            fps = 0
            display_time = time.time()
            self._stop = False
            while True:
                if self._stop:
                    logger.info("stop loop event")
                    return
                try:
                    self._loop_on_camera_events(q_nn, q_rgb, q_disparity)
                # pylint: disable=broad-except # bad frame or event must not stop loop
                except Exception as ex:
                    logger.exception("unexpected error: %s", str(ex))

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()
                if (time.time() - display_time) >= 10:
                    display_time = time.time()
                    logger.info("fps: %s", fps)

    def _loop_on_camera_events(self, q_nn: dai.DataOutputQueue, q_rgb: dai.DataOutputQueue, q_disparity: dai.DataOutputQueue) -> None:
        logger.debug("wait for new frame")

        # Wait for frame
        in_rgb: dai.ImgFrame = q_rgb.get()  # type: ignore # blocking call, will wait until a new data has arrived
        try:
            logger.debug("process frame")
            frame_ref = self._frame_processor.process(in_rgb)
        except FrameProcessError as ex:
            logger.error("unable to process frame: %s", str(ex))
            return
        logger.debug("frame processed")

        logger.debug("wait for nn response")
        # Read NN result
        in_nn: dai.NNData = q_nn.get()  # type: ignore
        logger.debug("process objects")
        self._object_processor.process(in_nn, frame_ref)
        logger.debug("objects processed")

        logger.debug("process disparity")
        if self._disparity_processor is not None:
            in_disparity: dai.ImgFrame = q_disparity.get()  # type: ignore
            self._disparity_processor.process(in_disparity, frame_ref=frame_ref,
                                              focal_length_in_pixels=self._focal_length_in_pixels)
        logger.debug("disparity processed")

    def stop(self) -> None:
        """
        Stop event loop, if loop is not running, do nothing
        :return:
        """
        self._stop = True


def _bbox_to_object(bbox: npt.NDArray[np.float64], score: float) -> evt.Object:
    obj = evt.Object()
    obj.type = evt.TypeObject.ANY
    obj.top = bbox[0].astype(float)
    obj.right = bbox[3].astype(float)
    obj.bottom = bbox[2].astype(float)
    obj.left = bbox[1].astype(float)
    obj.confidence = score
    return obj
