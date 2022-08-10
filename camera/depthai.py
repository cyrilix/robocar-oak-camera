import datetime
import logging
import paho.mqtt.client as mqtt

import events.events_pb2

import depthai as dai
import cv2

logger = logging.getLogger(__name__)


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_planar(frame):
    return frame.transpose(2, 0, 1).flatten()


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

        version = "2021.2"
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

        # colorCam = pipeline.create(dai.node.ColorCamera)
        # colorCam.setPreviewSize(256, 256)
        # colorCam.setVideoSize(1024, 1024)  # 4 times larger in both axis
        # colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # colorCam.setInterleaved(False)
        # colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # colorCam.setFps(10)
        #
        # controlIn = pipeline.create(dai.node.XLinkIn)
        # controlIn.setStreamName('control')
        # controlIn.out.link(colorCam.inputControl)
        #
        # cam_xout = pipeline.create(dai.node.XLinkOut)
        # cam_xout.setStreamName('video')
        # colorCam.video.link(cam_xout.input)

        # ---------------------------------------
        # 1st stage NN - text-detection
        # ---------------------------------------

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(
            blobconverter.from_zoo(name="east_text_detection_256x256", zoo_type="depthai", shaves=6, version=version))
        colorCam.preview.link(nn.input)

        nn_xout = pipeline.create(dai.node.XLinkOut)
        nn_xout.setStreamName('detections')
        nn.out.link(nn_xout.input)

        # ---------------------------------------
        # 2nd stage NN - text-recognition-0012
        # ---------------------------------------

        manip = pipeline.create(dai.node.ImageManip)
        manip.setWaitForConfigInput(True)

        manip_img = pipeline.create(dai.node.XLinkIn)
        manip_img.setStreamName('manip_img')
        manip_img.out.link(manip.inputImage)

        manip_cfg = pipeline.create(dai.node.XLinkIn)
        manip_cfg.setStreamName('manip_cfg')
        manip_cfg.out.link(manip.inputConfig)

        manip_xout = pipeline.create(dai.node.XLinkOut)
        manip_xout.setStreamName('manip_out')

        nn2 = pipeline.create(dai.node.NeuralNetwork)
        nn2.setBlobPath(blobconverter.from_zoo(name="text-recognition-0012", shaves=6, version=version))
        nn2.setNumInferenceThreads(2)
        manip.out.link(nn2.input)
        manip.out.link(manip_xout.input)

        nn2_xout = pipeline.create(dai.node.XLinkOut)
        nn2_xout.setStreamName("recognitions")
        nn2.out.link(nn2_xout.input)







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

        with dai.Device(self._pipeline) as device:
            q_vid = device.getOutputQueue("video", 4, blocking=False)
            # This should be set to block, but would get to some extreme queuing/latency!
            q_det = device.getOutputQueue("detections", 4, blocking=False)

            q_rec = device.getOutputQueue("recognitions", 4, blocking=True)

            q_manip_img = device.getInputQueue("manip_img")
            q_manip_cfg = device.getInputQueue("manip_cfg")
            q_manip_out = device.getOutputQueue("manip_out", 4, blocking=False)

            controlQueue = device.getInputQueue('control')

            frame = None
            cropped_stacked = None
            rotated_rectangles = []
            rec_pushed = 0
            rec_received = 0
            host_sync = HostSeqSync()

            characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
            codec = CTCCodec(characters)

            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)

            while True:
                vid_in = q_vid.tryGet()
                if vid_in is not None:
                    host_sync.add_msg(vid_in)

                # Multiple recognition results may be available, read until queue is empty
                while True:
                    in_rec = q_rec.tryGet()
                    if in_rec is None:
                        break
                    rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30, 1, 37)
                    decoded_text = codec.decode(rec_data)[0]
                    pos = rotated_rectangles[rec_received]
                    print("{:2}: {:20}".format(rec_received, decoded_text),
                          "center({:3},{:3}) size({:3},{:3}) angle{:5.1f} deg".format(
                              int(pos[0][0]), int(pos[0][1]), pos[1][0], pos[1][1], pos[2]))
                    # Draw the text on the right side of 'cropped_stacked' - placeholder
                    if cropped_stacked is not None:
                        cv2.putText(cropped_stacked, decoded_text,
                                    (120 + 10, 32 * rec_received + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('cropped_stacked', cropped_stacked)
                    rec_received += 1

                if cv2.waitKey(1) == ord('q'):
                    break

                if rec_received >= rec_pushed:
                    in_det = q_det.tryGet()
                    if in_det is not None:
                        frame = host_sync.get_msg(in_det.getSequenceNum()).getCvFrame().copy()

                        scores, geom1, geom2 = to_tensor_result(in_det).values()
                        scores = np.reshape(scores, (1, 1, 64, 64))
                        geom1 = np.reshape(geom1, (1, 4, 64, 64))
                        geom2 = np.reshape(geom2, (1, 1, 64, 64))

                        bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
                        boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
                        rotated_rectangles = [
                            east.get_cv_rotated_rect(bbox, angle * -1)
                            for (bbox, angle) in zip(boxes, angles)
                        ]

                        rec_received = 0
                        rec_pushed = len(rotated_rectangles)
                        if rec_pushed:
                            print("====== Pushing for recognition, count:", rec_pushed)
                        cropped_stacked = None
                        for idx, rotated_rect in enumerate(rotated_rectangles):
                            # Detections are done on 256x256 frames, we are sending back 1024x1024
                            # That's why we multiply center and size values by 4
                            rotated_rect[0][0] = rotated_rect[0][0] * 4
                            rotated_rect[0][1] = rotated_rect[0][1] * 4
                            rotated_rect[1][0] = rotated_rect[1][0] * 4
                            rotated_rect[1][1] = rotated_rect[1][1] * 4

                            # Draw detection crop area on input frame
                            points = np.int0(cv2.boxPoints(rotated_rect))
                            print(rotated_rect)
                            cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1,
                                          lineType=cv2.LINE_8)

                            # TODO make it work taking args like in OpenCV:
                            # rr = ((256, 256), (128, 64), 30)
                            rr = dai.RotatedRect()
                            rr.center.x = rotated_rect[0][0]
                            rr.center.y = rotated_rect[0][1]
                            rr.size.width = rotated_rect[1][0]
                            rr.size.height = rotated_rect[1][1]
                            rr.angle = rotated_rect[2]
                            cfg = dai.ImageManipConfig()
                            cfg.setCropRotatedRect(rr, False)
                            cfg.setResize(120, 32)
                            # Send frame and config to device
                            if idx == 0:
                                w, h, c = frame.shape
                                imgFrame = dai.ImgFrame()
                                imgFrame.setData(to_planar(frame))
                                imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                                imgFrame.setWidth(w)
                                imgFrame.setHeight(h)
                                q_manip_img.send(imgFrame)
                            else:
                                cfg.setReusePreviousImage(True)
                            q_manip_cfg.send(cfg)

                            # Get manipulated image from the device
                            transformed = q_manip_out.get().getCvFrame()

                            rec_placeholder_img = np.zeros((32, 200, 3), np.uint8)
                            transformed = np.hstack((transformed, rec_placeholder_img))
                            if cropped_stacked is None:
                                cropped_stacked = transformed
                            else:
                                cropped_stacked = np.vstack((cropped_stacked, transformed))

                if cropped_stacked is not None:
                    cv2.imshow('cropped_stacked', cropped_stacked)

                if frame is not None:
                    cv2.imshow('frame', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    print("Autofocus trigger (and disable continuous)")
                    ctrl = dai.CameraControl()
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                    ctrl.setAutoFocusTrigger()
                    controlQueue.send(ctrl)



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
