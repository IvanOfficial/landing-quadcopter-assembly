#!/usr/bin/env python

import time
#import 
# from utils.constants import XimeaParams
from xicamdriver import XiCamDriver


class StereoXiCamDriver:
    def __init__(self, left_serial_number, left_is_master,
                 right_serial_number, right_is_master, namespace="/stereo"):
        self.l_cam = XiCamDriver(serial_number=left_serial_number,
                                 master=left_is_master)
        self.r_cam = XiCamDriver(serial_number=right_serial_number,
                                 master=right_is_master)
        self.cams = self.l_cam, self.r_cam
        # TODO @maxtar how to remove this sleep time?
        time.sleep(2)
        self.make_stereo()
        for cam in self.cams:
            cam.start()

    def trig(self):
        for cam in self.cams:
            if cam.master is True:
                cam.trig()

    def make_stereo(self):
        master_is_set = False
        slave_is_set = False
        assert (len(self.cams) == 2), "There must be strictly two cameras."
        for cam in self.cams:
            if cam.master is True and master_is_set is False:
                cam.cam.set_trigger_source("XI_TRG_SOFTWARE")
                cam.cam.set_gpo_selector("XI_GPO_PORT1")
                cam.cam.set_gpo_mode("XI_GPO_FRAME_ACTIVE_NEG")  # XI_GPO_FRAME_ACTIVE_NEG / XI_GPO_EXPOSURE_ACTIVE
            elif cam.master is False and slave_is_set is False:
                cam.cam.set_trigger_source("XI_TRG_EDGE_RISING")
                cam.cam.set_gpi_selector("XI_GPI_PORT2")
                cam.cam.set_gpi_mode("XI_GPI_TRIGGER")
            elif (cam.master is True & master_is_set is True) or \
                    (cam.master is False & slave_is_set is True):
                raise Exception("One of the cameras must be a master and the other a slave.")

    def publish(self):
        for cam in self.cams:
            cam.publish(stamp)

    def stop(self):
        for cam in self.cams:
            cam.stop()
            
            
stereocam = StereoXiCamDriver(left_serial_number = 1, left_is_master = True,
                 right_serial_number = 2 , right_is_master = False)

while True:
    stereocam.publish()