"""
Mqtt gateway for oak-lite device
"""
import argparse
import logging
import os
import signal
import types
import typing
from typing import List

import depthai as dai
import paho.mqtt.client as mqtt

from camera import oak_pipeline as cam
from camera.oak_pipeline import StereoDepthPostFilter, MedianFilter, SpeckleFilter, TemporalFilter, SpatialFilter, \
    ThresholdFilter, DecimationFilter

CAMERA_EXPOSITION_DEFAULT = "default"
CAMERA_EXPOSITION_8300US = "8300us"
CAMERA_EXPOSITION_500US = "500us"

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
    parser.add_argument("-d", "---mqtt-topic-robocar-disparity",
                        help="MQTT topic where to publish disparity results",
                        default=_get_env_value("MQTT_TOPIC_DISPARITY", "/disparity"))
    parser.add_argument("-f", "--camera-fps",
                        help="set rate at which camera should produce frames",
                        type=int,
                        default=30)
    parser.add_argument("--camera-tuning-exposition", type=str,
                        default=CAMERA_EXPOSITION_DEFAULT,
                        help="override camera exposition configuration",
                        choices=[CAMERA_EXPOSITION_DEFAULT, CAMERA_EXPOSITION_500US, CAMERA_EXPOSITION_8300US])
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

    parser.add_argument("--disable-disparity", action="store_true",
                    help="enable disparity frame",
                    default=False
                    )
    parser.add_argument("--stereo-mode-lr-check",
                        help="remove incorrectly calculated disparity pixels due to occlusions at object borders",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-mode-extended-disparity",
                        help="allows detecting closer distance objects for the given baseline. This increases the maximum disparity search from 96 to 191, meaning the range is now: [0..190]",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-mode-subpixel",
                        help="iimproves the precision and is especially useful for long range measurements",
                        default=False, action="store_true"
                        )


    parser.add_argument("--stereo-post-processing-median-filter",
                        help="enable post-processing median filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-median-value",
                        help="Median filter config ",
                        type=str,
                        choices=["MEDIAN_OFF", "KERNEL_3x3", "KERNEL_5x5", "KERNEL_7x7"],
                        default="KERNEL_7x7",
                        )
    parser.add_argument("--stereo-post-processing-speckle-filter",
                        help="enable post-processing speckle filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-speckle-enable",
                        help="enable post-processing speckle filter",
                        type=bool, default=False
                        )
    parser.add_argument("--stereo-post-processing-speckle-range",
                        help="Speckle search range",
                        type=int, default=50
                        )

    parser.add_argument("--stereo-post-processing-temporal-filter",
                        help="enable post-processing temporal filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-temporal-persistency-mode",
                        help="Persistency mode.",
                        type=str, default="VALID_2_IN_LAST_4",
                        choices=["PERSISTENCY_OFF", "VALID_8_OUT_OF_8", "VALID_2_IN_LAST_3", "VALID_2_IN_LAST_4",
                                 "VALID_2_OUT_OF_8", "VALID_1_IN_LAST_2", "VALID_1_IN_LAST_5", "VALID_1_IN_LAST_8",
                                 "PERSISTENCY_INDEFINITELY"]
                        )
    parser.add_argument("--stereo-post-processing-temporal-alpha",
                        help="The Alpha factor in an exponential moving average with Alpha=1 - no filter. "
                             "Alpha = 0 - infinite filter. Determines the extent of the temporal history that should be "
                             "averaged. ",
                        type=float, default=0.4,
                        )
    parser.add_argument("--stereo-post-processing-temporal-delta",
                        help="Step-size boundary. Establishes the threshold used to preserve surfaces (edges). "
                             "If the disparity value between neighboring pixels exceed the disparity threshold set by "
                             "this delta parameter, then filtering will be temporarily disabled. Default value 0 means "
                             "auto: 3 disparity integer levels. In case of subpixel mode it’s 3*number of subpixel "
                             "levels.",
                        type=int, default=0,
                        )

    parser.add_argument("--stereo-post-processing-spatial-filter",
                        help="enable post-processing spatial filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-spatial-enable",
                        help="Whether to enable or disable the filter",
                        type=bool, default=False,
                        )
    parser.add_argument("--stereo-post-processing-spatial-hole-filling-radius",
                        help="An in-place heuristic symmetric hole-filling mode applied horizontally during the filter passes",
                        type=int, default=2,
                        )
    parser.add_argument("--stereo-post-processing-spatial-alpha",
                        help="The Alpha factor in an exponential moving average with Alpha=1 - no filter. Alpha = 0 - infinite filter",
                        type=float, default=0.5,
                        )
    parser.add_argument("--stereo-post-processing-spatial-delta",
                        help="Step-size boundary. Establishes the threshold used to preserve edges",
                        type=int, default=0,
                        )
    parser.add_argument("--stereo-post-processing-spatial-num-iterations",
                        help="Number of iterations over the image in both horizontal and vertical direction",
                        type=int, default=1,
                        )

    parser.add_argument("--stereo-post-processing-threshold-filter",
                        help="enable post-processing threshold filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-threshold-min-range",
                        help="Minimum range in depth units. Depth values under this value are invalidated",
                        type=int, default=500,
                        )
    parser.add_argument("--stereo-post-processing-threshold-max-range",
                        help="Maximum range in depth units. Depth values over this value are invalidated.",
                        type=int, default=15000,
                        )

    parser.add_argument("--stereo-post-processing-decimation-filter",
                        help="enable post-processing decimation filter",
                        default=False, action="store_true"
                        )
    parser.add_argument("--stereo-post-processing-decimation-decimal-factor",
                        help="Decimation factor",
                        type=int, default=1, choices=[1, 2, 3, 4]
                        )
    parser.add_argument("--stereo-post-processing-decimation-mode",
                        help="Decimation algorithm type",
                        type=str, default="PIXEL_SKIPPING",
                        choices=["PIXEL_SKIPPING", "NON_ZERO_MEDIAN", "NON_ZERO_MEAN"]
                        )

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
    if args.disable_disparity == False:
        depth_source = cam.DepthSource(pipeline=pipeline,
                                       extended_disparity=args.stereo_mode_extended_disparity,
                                       subpixel=args.stereo_mode_subpixel,
                                       lr_check=args.stereo_mode_lr_check,
                                       stereo_filters=stereo_filters),
        disparity_processor = cam.DisparityProcessor(mqtt_client=client, disparity_topic=args.mqtt_topic_robocar_disparity)
    else:
        disparity_processor = None
        depth_source = None

    pipeline = dai.Pipeline()
    if args.camera_tuning_exposition == CAMERA_EXPOSITION_500US:
        pipeline.setCameraTuningBlobPath('/camera_tuning/tuning_exp_limit_500us.bin')
    elif args.camera_tuning_exposition == CAMERA_EXPOSITION_8300US:
        pipeline.setCameraTuningBlobPath('/camera_tuning/tuning_exp_limit_8300us.bin')

    stereo_filters = _get_stereo_filters(args)

    pipeline_controller = cam.PipelineController(pipeline=pipeline,
                                                 frame_processor=frame_processor,
                                                 object_processor=object_processor,
                                                 object_node=cam.ObjectDetectionNN(pipeline=pipeline),
                                                 camera=cam.CameraSource(pipeline=pipeline,
                                                                         img_width=args.image_width,
                                                                         img_height=args.image_height,
                                                                         fps=args.camera_fps,
                                                                         ),
                                                 depth_source=depth_source,
                                                 disparity_processor=disparity_processor)

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


def _get_stereo_filters(args: argparse.Namespace) -> List[StereoDepthPostFilter]:
    filters = []

    if args.stereo_post_processing_median_filter:
        if args.stereo_post_processing_median_value == "MEDIAN_OFF":
            value = dai.MedianFilter.MEDIAN_OFF
        elif args.stereo_post_processing_median_value == "KERNEL_3x3":
            value = dai.MedianFilter.KERNEL_3x3
        elif args.stereo_post_processing_median_value == "KERNEL_5x5":
            value = dai.MedianFilter.KERNEL_5x5
        elif args.stereo_post_processing_median_value == "KERNEL_7x7":
            value = dai.MedianFilter.KERNEL_7x7
        else:
            value = dai.MedianFilter.KERNEL_7x7

        filters.append(MedianFilter(value=value))

    if args.stereo_post_processing_speckle_filter:
        filters.append(SpeckleFilter(enable=args.stereo_post_processing_speckle_enable,
                                     speckle_range=args.stereo_post_processing_speckle_range))

    if args.stereo_post_processing_temporal_filter:
        if args.stereo_post_processing_temporal_persistency-mode == "PERSISTENCY_OFF":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_OFF
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_8_OUT_OF_8":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_2_IN_LAST_3":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_2_IN_LAST_4":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_2_OUT_OF_8":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_OUT_OF_8
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_1_IN_LAST_2":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_2
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_1_IN_LAST_5":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_5
        elif args.stereo_post_processing_temporal_persistency-mode == "VALID_1_IN_LAST_8":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_8
        elif args.stereo_post_processing_temporal_persistency-mode == "PERSISTENCY_INDEFINITELY":
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_INDEFINITELY
        else:
            mode=dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4

        filters.append(TemporalFilter(
            enable=args.stereo_post_processing_temporal_enable,
            persistencyMode=mode,
            alpha=args.stereo_post_processing_temporal_alpha,
            delta=args.stereo_post_processing_temporal_delta
        ))

        if args.stereo_post_processing_spatial_filter:
            filters.append(SpatialFilter(enable=args.stereo_post_processing_spatial_enable,
                                         hole_filling_radius=args.stereo_post_processing_spatial_hole_filling_radius,
                                         alpha=args.stereo_post_processing_spatial_alpha,
                                         delta=args.stereo_post_processing_spatial_delta,
                                         num_iterations=args.stereo_post_processing_spatial_num_iterations,
                                         ))

        if args.stereo_post_processing_threshold_filter:
            filters.append(ThresholdFilter(
                min_range=args.stereo_post_processing_threshold_min_range,
                max_range=args.stereo_post_processing_threshold_max_range,
            ))

        if args.stereo_post_processing_decimation_filter:

            if args.stereo_post_processing_decimation_mode == "PIXEL_SKIPPING":
                mode=dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING
            if args.stereo_post_processing_decimation_mode == "NON_ZERO_MEDIAN":
                mode=dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
            if args.stereo_post_processing_decimation_mode == "NON_ZERO_MEAN":
                mode=dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN
            else:
                mode=dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING

            filters.append(DecimationFilter(
                decimation_factor=args.stereo_post_processing_decimation_decimal_factor,
                mode=mode
            ))

    return filters

if __name__ == '__main__':
    execute_from_command_line()