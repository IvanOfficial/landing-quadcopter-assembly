from calibrovka_class import  stereo_calibrator
name_left = "25797059"
name_right = "25791059"
calib = stereo_calibrator(name_left, name_right, 8, 8)
calib.stereo_calibration()