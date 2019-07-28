#!/usr/bin/env python

import cv2

from ximea import xiapi
from constants import XimeaParams

class XiCamDriver(XimeaParams):
    # Third parameter is for stereo vision,
    # it's boolean and means master (if True),
    # slave (if False) or single camera (if None).
    #
    # Example URL syntax:
    # - `file:///full/path/to/local/file.yaml`
    # - `package://ros_package_name/calibrations/camera.yaml`
    #
    # The `file:` URL specifies a full path name in the local system.
    # The `package:` URL is handled the same as `file:`, except the path
    # name is resolved relative to the location of the named ROS
    # package, which must be reachable via `$ROS_PACKAGE_PATH`.
    # TODO @maxtar move strings to the constants class
    def __init__(self, serial_number, master="None"):

        self.cam = xiapi.Camera()
        self.img = xiapi.Image()
        self.master = master
        self._init_camera(serial_number)
        self.serial_number = serial_number

    def _init_camera(self, serial_number):
        self.cam.open_device_by_SN(serial_number)
        self._set_params()

    def start(self):
        self.cam.start_acquisition()

    def publish(self, timestamp=None):

        self.get_image()
        image_data_numpy = self.img.get_image_data_numpy()
        np.save("./images/image" + str(self.serial_number), image_data_numpy)
    def _set_params(self,
                    img_data_format=XimeaParams.IMG_DATA_FORMAT,
                    exposure=XimeaParams.EXPOSURE,
                    gain=XimeaParams.GAIN,
                    downsampling_type=XimeaParams.DOWNSAMPLING_TYPE,
                    downsampling_rate=XimeaParams.DOWNSAMPLING_RATE):
        self.cam.set_imgdataformat(img_data_format)
        self.cam.set_exposure(exposure)
        self.cam.set_gain(gain)

        self.cam.set_downsampling_type("XI_" + downsampling_type)
        self.cam.set_downsampling("XI_DWN_" + downsampling_rate)

    def get_image(self):
        self.cam.get_image(self.img)
        # print("\nIMG_TIMESTAMP: " + str(self.img.tsSec) + "." + str(self.img.tsUSec) + "\n")
        # print("\nCAM_TIMESTAMP: " + str(self.cam.get_timestamp()) + "\n")
        return

    def trig(self):
        self.cam.set_trigger_software(1)

    def stop(self):
        self.cam.stop_acquisition()
        self.cam.close_device()
