import datetime
import logging
import paho.mqtt.client as mqtt

import events.events_pb2

import depthai as dai
import cv2
import numpy as np

logger = logging.getLogger(__name__)

NN_PATH = "/models/mobile_object_localizer_192x192_openvino_2021.4_6shave.blob"
NN_WIDTH = 192
NN_HEIGHT = 192


class FramePublisher:
    def __init__(self, mqtt_client: mqtt.Client, frame_topic: str, objects_topic: str, objects_threshold: float,
                 img_width: int, img_height: int):
        self._mqtt_client = mqtt_client
        self._frame_topic = frame_topic
        self._objects_topic = objects_topic
        self._objects_threshold = objects_threshold
        self._img_width = img_width
        self._img_height = img_height
        self._pipeline = self._configure_pipeline()

    def _configure_pipeline(self) -> dai.Pipeline:
        logger.info("configure pipeline")
        pipeline = dai.Pipeline()

        pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(NN_PATH)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        xout_nn.input.setBlocking(False)

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
            q_rgb = device.getOutputQueue(name="rgb", maxSize=queue_size, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=queue_size, blocking=False)

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

                    in_nn = q_nn.get()

                    # get outputs
                    detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
                    detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))

                    # keep boxes bigger than threshold
                    mask = detection_scores >= self._objects_threshold
                    boxes = detection_boxes[mask]
                    scores = detection_scores[mask]

                    if boxes.shape[0] > 0:
                        objects_msg = events.events_pb2.ObjectsMessage()
                        objs = []
                        for i in range(boxes.shape[0]):
                            bbox = boxes[i]
                            logger.debug("new object detected: %s", str(bbox))
                            o = events.events_pb2.Object()
                            o.type = events.events_pb2.TypeObject.ANY
                            o.top = bbox[0].astype(float)
                            o.right = bbox[3].astype(float)
                            o.bottom = bbox[2].astype(float)
                            o.left = bbox[1].astype(float)
                            o.confidence = scores[i].astype(float)
                            objs.append(o)
                        objects_msg.objects.extend(objs)

                        objects_msg.frame_ref.name = frame_msg.id.name
                        objects_msg.frame_ref.id = frame_msg.id.id
                        objects_msg.frame_ref.created_at.FromDatetime(now)

                        logger.debug("publish object event to %s", self._frame_topic)
                        self._mqtt_client.publish(topic=self._objects_topic,
                                                  payload=objects_msg.SerializeToString(),
                                                  qos=0,
                                                  retain=False)

                except Exception as e:
                    logger.exception("unexpected error: %s", str(e))
