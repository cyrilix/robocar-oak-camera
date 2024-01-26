# robocar-oak-camera

Mqtt gateway for oak-lite device
  
## Docker

To build images, run script:

```bash 
./build-docker.sh
```

## Usage

```shell
usage: cli.py [-h] [-u MQTT_USERNAME] [-p MQTT_PASSWORD] [-b MQTT_BROKER_HOST]
              [-P MQTT_BROKER_PORT] [-C MQTT_CLIENT_ID]
              [-c MQTT_TOPIC_ROBOCAR_OAK_CAMERA]
              [-o MQTT_TOPIC_ROBOCAR_OBJECTS] [-t OBJECTS_THRESHOLD]
              [-d MQTT_TOPIC_ROBOCAR_DISPARITY] [-f CAMERA_FPS]
              [--camera-tuning-exposition {default,500us,8300us}]
              [-H IMAGE_HEIGHT] [-W IMAGE_WIDTH] [--log {info,debug}]
              [--stereo-mode-lr-check] [--stereo-mode-extended-disparity]
              [--stereo-mode-subpixel]
              [--stereo-post-processing-median-filter]
              [--stereo-post-processing-median-value {MEDIAN_OFF,KERNEL_3x3,KERNEL_5x5,KERNEL_7x7}]
              [--stereo-post-processing-speckle-filter]
              [--stereo-post-processing-speckle-enable STEREO_POST_PROCESSING_SPECKLE_ENABLE]
              [--stereo-post-processing-speckle-range STEREO_POST_PROCESSING_SPECKLE_RANGE]
              [--stereo-post-processing-temporal-filter]
              [--stereo-post-processing-temporal-persistency-mode {PERSISTENCY_OFF,VALID_8_OUT_OF_8,VALID_2_IN_LAST_3,VALID_2_IN_LAST_4,VALID_2_OUT_OF_8,VALID_1_IN_LAST_2,VALID_1_IN_LAST_5,VALID_1_IN_LAST_8,PERSISTENCY_INDEFINITELY}]
              [--stereo-post-processing-temporal-alpha STEREO_POST_PROCESSING_TEMPORAL_ALPHA]
              [--stereo-post-processing-temporal-delta STEREO_POST_PROCESSING_TEMPORAL_DELTA]
              [--stereo-post-processing-spatial-filter]
              [--stereo-post-processing-spatial-enable STEREO_POST_PROCESSING_SPATIAL_ENABLE]
              [--stereo-post-processing-spatial-hole-filling-radius STEREO_POST_PROCESSING_SPATIAL_HOLE_FILLING_RADIUS]
              [--stereo-post-processing-spatial-alpha STEREO_POST_PROCESSING_SPATIAL_ALPHA]
              [--stereo-post-processing-spatial-delta STEREO_POST_PROCESSING_SPATIAL_DELTA]
              [--stereo-post-processing-spatial-num-iterations STEREO_POST_PROCESSING_SPATIAL_NUM_ITERATIONS]
              [--stereo-post-processing-threshold-filter]
              [--stereo-post-processing-threshold-min-range STEREO_POST_PROCESSING_THRESHOLD_MIN_RANGE]
              [--stereo-post-processing-threshold-max-range STEREO_POST_PROCESSING_THRESHOLD_MAX_RANGE]
              [--stereo-post-processing-decimation-filter]
              [--stereo-post-processing-decimation-decimal-factor {1,2,3,4}]
              [--stereo-post-processing-decimation-mode {PIXEL_SKIPPING,NON_ZERO_MEDIAN,NON_ZERO_MEAN}]

options:
  -h, --help            show this help message and exit
  -u MQTT_USERNAME, --mqtt-username MQTT_USERNAME
                        MQTT user
  -p MQTT_PASSWORD, --mqtt-password MQTT_PASSWORD
                        MQTT password
  -b MQTT_BROKER_HOST, --mqtt-broker-host MQTT_BROKER_HOST
                        MQTT broker host
  -P MQTT_BROKER_PORT, --mqtt-broker-port MQTT_BROKER_PORT
                        MQTT broker port
  -C MQTT_CLIENT_ID, --mqtt-client-id MQTT_CLIENT_ID
                        MQTT client id
  -c MQTT_TOPIC_ROBOCAR_OAK_CAMERA, --mqtt-topic-robocar-oak-camera MQTT_TOPIC_ROBOCAR_OAK_CAMERA
                        MQTT topic where to publish robocar-oak-camera frames
  -o MQTT_TOPIC_ROBOCAR_OBJECTS, ---mqtt-topic-robocar-objects MQTT_TOPIC_ROBOCAR_OBJECTS
                        MQTT topic where to publish objects detection results
  -t OBJECTS_THRESHOLD, --objects-threshold OBJECTS_THRESHOLD
                        threshold to filter detected objects
  -d MQTT_TOPIC_ROBOCAR_DISPARITY, ---mqtt-topic-robocar-disparity MQTT_TOPIC_ROBOCAR_DISPARITY
                        MQTT topic where to publish disparity results
  -f CAMERA_FPS, --camera-fps CAMERA_FPS
                        set rate at which camera should produce frames
  --camera-tuning-exposition {default,500us,8300us}
                        override camera exposition configuration
  -H IMAGE_HEIGHT, --image-height IMAGE_HEIGHT
                        image height
  -W IMAGE_WIDTH, --image-width IMAGE_WIDTH
                        image width
  --log {info,debug}    Log level
  --stereo-mode-lr-check
                        remove incorrectly calculated disparity pixels due to
                        occlusions at object borders
  --stereo-mode-extended-disparity
                        allows detecting closer distance objects for the given
                        baseline. This increases the maximum disparity search
                        from 96 to 191, meaning the range is now: [0..190]
  --stereo-mode-subpixel
                        iimproves the precision and is especially useful for
                        long range measurements
  --stereo-post-processing-median-filter
                        enable post-processing median filter
  --stereo-post-processing-median-value {MEDIAN_OFF,KERNEL_3x3,KERNEL_5x5,KERNEL_7x7}
                        Median filter config
  --stereo-post-processing-speckle-filter
                        enable post-processing speckle filter
  --stereo-post-processing-speckle-enable STEREO_POST_PROCESSING_SPECKLE_ENABLE
                        enable post-processing speckle filter
  --stereo-post-processing-speckle-range STEREO_POST_PROCESSING_SPECKLE_RANGE
                        Speckle search range
  --stereo-post-processing-temporal-filter
                        enable post-processing temporal filter
  --stereo-post-processing-temporal-persistency-mode {PERSISTENCY_OFF,VALID_8_OUT_OF_8,VALID_2_IN_LAST_3,VALID_2_IN_LAST_4,VALID_2_OUT_OF_8,VALID_1_IN_LAST_2,VALID_1_IN_LAST_5,VALID_1_IN_LAST_8,PERSISTENCY_INDEFINITELY}
                        Persistency mode.
  --stereo-post-processing-temporal-alpha STEREO_POST_PROCESSING_TEMPORAL_ALPHA
                        The Alpha factor in an exponential moving average with
                        Alpha=1 - no filter. Alpha = 0 - infinite filter.
                        Determines the extent of the temporal history that
                        should be averaged.
  --stereo-post-processing-temporal-delta STEREO_POST_PROCESSING_TEMPORAL_DELTA
                        Step-size boundary. Establishes the threshold used to
                        preserve surfaces (edges). If the disparity value
                        between neighboring pixels exceed the disparity
                        threshold set by this delta parameter, then filtering
                        will be temporarily disabled. Default value 0 means
                        auto: 3 disparity integer levels. In case of subpixel
                        mode it’s 3*number of subpixel levels.
  --stereo-post-processing-spatial-filter
                        enable post-processing spatial filter
  --stereo-post-processing-spatial-enable STEREO_POST_PROCESSING_SPATIAL_ENABLE
                        Whether to enable or disable the filter
  --stereo-post-processing-spatial-hole-filling-radius STEREO_POST_PROCESSING_SPATIAL_HOLE_FILLING_RADIUS
                        An in-place heuristic symmetric hole-filling mode
                        applied horizontally during the filter passes
  --stereo-post-processing-spatial-alpha STEREO_POST_PROCESSING_SPATIAL_ALPHA
                        The Alpha factor in an exponential moving average with
                        Alpha=1 - no filter. Alpha = 0 - infinite filter
  --stereo-post-processing-spatial-delta STEREO_POST_PROCESSING_SPATIAL_DELTA
                        Step-size boundary. Establishes the threshold used to
                        preserve edges
  --stereo-post-processing-spatial-num-iterations STEREO_POST_PROCESSING_SPATIAL_NUM_ITERATIONS
                        Number of iterations over the image in both horizontal
                        and vertical direction
  --stereo-post-processing-threshold-filter
                        enable post-processing threshold filter
  --stereo-post-processing-threshold-min-range STEREO_POST_PROCESSING_THRESHOLD_MIN_RANGE
                        Minimum range in depth units. Depth values under this
                        value are invalidated
  --stereo-post-processing-threshold-max-range STEREO_POST_PROCESSING_THRESHOLD_MAX_RANGE
                        Maximum range in depth units. Depth values over this
                        value are invalidated.
  --stereo-post-processing-decimation-filter
                        enable post-processing decimation filter
  --stereo-post-processing-decimation-decimal-factor {1,2,3,4}
                        Decimation factor
  --stereo-post-processing-decimation-mode {PIXEL_SKIPPING,NON_ZERO_MEDIAN,NON_ZERO_MEAN}
                        Decimation algorithm type

```