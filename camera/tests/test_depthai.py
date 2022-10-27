import datetime
import unittest.mock

import depthai as dai
import numpy as np
import paho.mqtt.client as mqtt
import pytest
import pytest_mock

import camera.depthai
import events.events_pb2

Object = dict[str, float]


@pytest.fixture
def mqtt_client(mocker: pytest_mock.MockerFixture) -> mqtt.Client:
    return mocker.MagicMock()


class TestObjectProcessor:
    @pytest.fixture
    def frame_ref(self):
        now = datetime.datetime.now()
        frame_msg = events.events_pb2.FrameMessage()
        frame_msg.id.name = "robocar-oak-camera-oak"
        frame_msg.id.id = str(int(now.timestamp() * 1000))
        frame_msg.id.created_at.FromDatetime(now)
        return frame_msg.id

    @pytest.fixture
    def object1(self) -> Object:
        return {
            "left": 0.3,
            "right": 0.7,
            "top": 0.1,
            "bottom": 0.6,
            "score": 0.8,
        }

    @pytest.fixture
    def raw_objects_empty(self, mocker: pytest_mock.MockerFixture) -> dai.NNData:
        raw_objects = mocker.MagicMock()

        def mock_return(name):
            if name == "ExpandDims":
                return [[0] * 4] * 100
            elif name == "ExpandDims_2":
                return [0] * 100
            else:
                raise ValueError(f"{name} is not a valid arg")

        m = mocker.patch(target='depthai.NNData.getLayerFp16', autospec=True)
        m.getLayerFp16 = mock_return
        return m

    @pytest.fixture
    def raw_objects_one(self, mocker: pytest_mock.MockerFixture, object1: Object) -> dai.NNData:
        def mock_return(name):
            if name == "ExpandDims":  # Detection boxes
                boxes: list[list[float]] = [[0.] * 4] * 100
                boxes[0] = [object1["top"], object1["left"], object1["bottom"], object1["right"]]
                return np.array(boxes)

            elif name == "ExpandDims_2":  # Detection scores
                scores: list[float] = [0.] * 100
                scores[0] = object1["score"]
                return scores
            else:
                raise ValueError(f"{name} is not a valid arg")

        m = mocker.patch(target='depthai.NNData.getLayerFp16', autospec=True)
        m.getLayerFp16 = mock_return
        return m

    @pytest.fixture
    def object_processor(self, mqtt_client: mqtt.Client) -> camera.depthai.ObjectProcessor:
        return camera.depthai.ObjectProcessor(mqtt_client, "topic/object", 0.2)

    def test_process_without_object(self, object_processor: camera.depthai.ObjectProcessor, mqtt_client,
                                    raw_objects_empty, frame_ref):
        object_processor.process(raw_objects_empty, frame_ref)
        mqtt_client.publish.assert_not_called()

    def test_process_with_object_with_low_score(self, object_processor: camera.depthai.ObjectProcessor, mqtt_client,
                                                raw_objects_one, frame_ref):
        object_processor._objects_threshold = 0.9
        object_processor.process(raw_objects_one, frame_ref)
        mqtt_client.publish.assert_not_called()

    def test_process_with_one_object(self,
                                     object_processor: camera.depthai.ObjectProcessor, mqtt_client,
                                     raw_objects_one, frame_ref, object1: Object):
        object_processor.process(raw_objects_one, frame_ref)
        left = object1["left"]
        right = object1["right"]
        top = object1["top"]
        bottom = object1["bottom"]
        score = object1["score"]

        pub_mock: unittest.mock.MagicMock = mqtt_client.publish
        pub_mock.assert_called_once_with(payload=unittest.mock.ANY, qos=0, retain=False, topic="topic/object")
        payload = pub_mock.call_args.kwargs['payload']
        objects_msg = events.events_pb2.ObjectsMessage()
        objects_msg.ParseFromString(payload)
        assert len(objects_msg.objects) == 1
        assert left - 0.0001 < objects_msg.objects[0].left < left + 0.0001
        assert right - 0.0001 < objects_msg.objects[0].right < right + 0.0001
        assert top - 0.0001 < objects_msg.objects[0].top < top + 0.0001
        assert bottom - 0.0001 < objects_msg.objects[0].bottom < bottom + 0.0001
        assert score - 0.0001 < objects_msg.objects[0].confidence < score + 0.0001
        assert objects_msg.frame_ref == frame_ref


class TestFrameProcessor:
    @pytest.fixture
    def frame_processor(self, mqtt_client: mqtt.Client) -> camera.depthai.FrameProcessor:
        return camera.depthai.FrameProcessor(mqtt_client, "topic/frame")

    def test_process(self, frame_processor: camera.depthai.FrameProcessor, mocker: pytest_mock.MockerFixture,
                     mqtt_client: mqtt.Client):
        img: dai.ImgFrame = mocker.MagicMock()
        mocker.patch(target="cv2.imencode").return_value = (True, np.array(b"img content"))

        frame_ref = frame_processor.process(img)

        pub_mock: unittest.mock.MagicMock = mqtt_client.publish
        pub_mock.assert_called_once_with(payload=unittest.mock.ANY, qos=0, retain=False, topic="topic/frame")
        payload = pub_mock.call_args.kwargs['payload']
        frame_msg = events.events_pb2.FrameMessage()
        frame_msg.ParseFromString(payload)

        assert frame_msg.id == frame_ref
        assert frame_msg.frame == b"img content"

        assert frame_msg.id.name == "robocar-oak-camera-oak"
        assert len(frame_msg.id.id) is 13
        now = datetime.datetime.now()
        assert now - datetime.timedelta(
            milliseconds=10) < frame_msg.id.created_at.ToDatetime() < now + datetime.timedelta(milliseconds=10)

    def test_process_error(self, frame_processor: camera.depthai.FrameProcessor, mocker: pytest_mock.MockerFixture,
                           mqtt_client: mqtt.Client):
        img: dai.ImgFrame = mocker.MagicMock()
        mocker.patch(target="cv2.imencode").return_value = (False, None)

        with pytest.raises(camera.depthai.FrameProcessError) as ex:
            _ = frame_processor.process(img)
        exception_raised = ex.value
        assert exception_raised.message == "unable to process to encode frame to jpg"
