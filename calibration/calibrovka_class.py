import cv2
import numpy as np
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tqdm import tqdm


class calibrator:
    def __init__(self, number_cam, fl):
        self.number_cam = number_cam
        self.ret = None
        self.K = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.focal_length = fl
        self.detected = None
        self.count = None
        self.line_to_images = './calibration_images/' + number_cam + '/*'
        self.line_to_params = "./camera_params/" + number_cam + "/"
        self.obj_points = None
        self.imagePoints1 = None
        self.gray_image_shape = None
        self.image_size = None
        self.detected_image_list = []
        self.line_to_images_new = './calibration_images/new/' + number_cam + '/*'

    def save_params(self):
        # Save parameters into numpy file
        line = self.line_to_params
        np.save(line + "ret", self.ret)
        np.save(line + "K", self.K)
        np.save(line + "dist", self.dist)
        np.save(line + "rvecs", self.rvecs)
        np.save(line + "tvecs", self.tvecs)
        np.save(line + "FocalLength", self.focal_length)

    def calibration_(self):
        # ============================================
        # Camera calibration
        # ============================================

        # Define size of chessboard target.

        chessboard_size = (6, 9)
        # chessboard_size = (7,5)
        # Define arrays to save detected points
        obj_points = []  # 3D points in real world space
        img_points = []  # 3D points in image plane

        # Prepare grid and points to display

        objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)

        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # read images
        line = self.line_to_images_new
        calibration_paths = glob.glob(line)
        t = 0
        y = 0
        # Iterate over images to find intrinsic matrix
        #for image_path in tqdm_notebook(calibration_paths):
        print("Calibration ..." + "\n" + "camera: " + self.number_cam)
        for image_path in tqdm(calibration_paths):

            # Load image
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # print("Image loaded, Analizying...")
            # find chessboard corners
            # plt.imshow(gray_image)
            # plt.show()
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

            if ret == True:
                # print("+++++++++++++++++Chessboard detected!+++++++++++++++++")
                # print(image_path)
                # define criteria for subpixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # refine corner location (to subpixel accuracy) based on criteria.
                cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)
                self.detected_image_list.append(image_path)
                y += 1
            t += 1
        h, w = image.shape[:2]
        self.image_size = (h,w)
        self.detected = y
        self.count = t
        # Calibrate camera
        self.ret, self.K, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(obj_points,
                                                                                  img_points,
                                                                                  gray_image.shape[::-1],
                                                                                  None, None)
        self.obj_points = obj_points
        self.img_points = img_points
        self.gray_image_shape = gray_image.shape[::-1]
        print("================ Calibration was successful  =================")

    def calibration_cam(self):
        self.calibration_()
        self.save_params()
        print("There were only images: " + str(self.count) + "\n" + "There were a total of images found: " + str(
            self.detected))


    def make_detected_image_list(self):

        # read images
        line = self.line_to_images
        calibration_paths = glob.glob(line)
        chessboard_size = (6, 9)
        print("Make detected_image_list, camera: " + self.number_cam)
        for image_path in tqdm(calibration_paths):

            # Load image
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

            if ret == True:
                self.detected_image_list.append(image_path)



class stereo_calibrator:
    def __init__(self, number_cam_left, number_cam_right, fl_left, fl_right):
        self.number_cam_left = number_cam_left
        self.number_cam_right = number_cam_right
        self.link_to_recreate = './calibration_images/new/'
        self.calibrovka_left = calibrator(number_cam_left, fl_left)
        self.calibrovka_right = calibrator(number_cam_right, fl_right)

    def stereo_calibration_separately(self):
        self.calibrovka_left.calibration_cam()
        self.calibrovka_right.calibration_cam()


    def stereo_calibration_together(self):
        ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=self.calibrovka_left.obj_points,
            imagePoints1=self.calibrovka_left.img_points,
            imagePoints2=self.calibrovka_right.img_points,
            cameraMatrix1=self.calibrovka_left.K,
            distCoeffs1=self.calibrovka_left.dist,
            cameraMatrix2=self.calibrovka_right.K,
            distCoeffs2=self.calibrovka_right.dist,
            imageSize=self.calibrovka_left.gray_image_shape)
        # see more obout this parameters : https://docs.opencv.org/2.4/modules/calib3d/doc/calib3d.html

        link = "./camera_params/stereo_params/"
        # Save parameters into numpy file
        np.save(link + "ret", ret)
        np.save(link + "K_left", K_left)
        np.save(link + "K_right", K_right)
        np.save(link + "dist_left", dist_left)
        np.save(link + "dist_right", dist_right)
        np.save(link + "R", R)
        np.save(link + "T", T)
        np.save(link + "E", E)
        np.save(link + "F", F)
        np.save(link + "image_size", self.calibrovka_left.image_size)
        print("================ Stereocalibration was successful  =================")

    def stereo_calibration(self):
    	self.selection()
    	#self.stereo_calibration_separately()
    	#self.stereo_calibration_together()
    	
    def re_create_data_images(self):
    	ll = self.calibrovka_left.detected_image_list
    	lr = self.calibrovka_right.detected_image_list
    	print("Re-create data images")
    	for l in tqdm(ll):
    		if l in lr:
    			image_left = cv2.imread(self.calibrovka_left.line_to_images + l)
    			image_right = cv2.imread(self.calibrovka_right.line_to_images + l)
    			cv2.imwrite(self.link_to_recreate + self.calibrovka_left.number_cam + '/' + l, image_left)
    			cv2.imwrite(self.link_to_recreate + self.calibrovka_right.number_cam + '/' + l, image_right)

    def selection(self):
    	self.calibrovka_left.make_detected_image_list()
    	self.calibrovka_right.make_detected_image_list()
    	self.re_create_data_images()

