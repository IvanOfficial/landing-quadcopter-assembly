#!/usr/bin/env python

# MAIN
DAMPING = 2  # Damping coefficient for velocity along the z axis (while takeoff)
DESIRED_SPEED = 5  # TODO comment
DH = 5  # Takeoff height radius (delta) (in meters)
FREQUENCY = 15  # Rospy rate frequency
HEIGHT = 9  # Takeoff height (in meters)

# CONVERTER
CDC = 4  # Converter discharging coefficient (converter puts every CDCth image to the queue)

# MATCHER
MAXVAL = 2500000  # Threshold maximum value (xx00000 format)

# CAMERA
FOCAL_LENGTH = 4.2e-3  # Focal length, depends on the lens (objective) (in meters)
HORIZONTAL_FOV = 1.146681  # Horizontal field of view, depends on the lens (objective) (in rads) // 0.986111
REAL_SIZE = 2  # Real size of an object (in meters)
RES_WIDTH = 2592  # Width of an image (in pixels)
RES_HEIGHT = 1944  # Height of an image (in pixels)
SENSOR_HEIGHT = 4.3e-3  # Sensor size, depends on sensor (matrix) (in meters)
SENSOR_RATIO = 1.33  # Sensor size is 5.7 mm x 4.3 mm (diagonal is 7.1 mm)
SENSOR_WIDTH = 5.7e-3  # Sensor size, depends on sensor (matrix) (in meters)

# ?
RTA = 1
KH = 0.4


# XIMEA
# noinspection PyClassHasNoInit
class XimeaParams:
    IMG_DATA_FORMAT = "XI_RGB24"  # "XI_RGB24" / "XI_MONO8"
    EXPOSURE = 150
    GAIN = 2
    DOWNSAMPLING_TYPE = "BINNING"  # "BINNING" / "SKIPPING"
    DOWNSAMPLING_RATE = "2x2"

    # noinspection PyClassHasNoInit
    class Stereo:
        RIGHT_SN = "25794659"
        LEFT_SN = "25790759"
        ROSPY_RATE = 30
