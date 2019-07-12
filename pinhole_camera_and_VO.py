#source: https://github.com/uoip/monoVO-python 
import numpy as np 
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

#lucas kanade(lk) optical flow
lk_params = dict(winSize  = (21, 21), 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                            30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, 
                                            **lk_params)  #shape: [k,2] [k,1] 
                                                            #[k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class PinholeCamera:
    """
    Attributes:
    fx, fy: focal lengths expressed as pixel units
    cx, cy: image principle points
    source: https://goo.gl/6LPrcX 
    
    """
    def __init__(self, width, height, fx, fy, cx, cy, d):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(d[0]) > 0.0000001)
        self.d = d


class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame, self.last_frame = None, None
        self.cur_R, self.cur_t = None, None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.X, self.Y, self.Z = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, 
                                                       nonmaxSuppression=True)


    def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry 
                                            #dataset
        #getting the pose for the previous frame
        pose = self.poses[frame_id-1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        #getting the pose for the current frame        
        pose = self.poses[frame_id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        #the absolute scale between the coordinates from the two frames
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + 
                       (z - z_prev)*(z - z_prev))

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, 
                                                   self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, 
                                       focal=self.focal, pp=self.pp, 
                                       method=cv2.RANSAC, prob=0.999, 
                                       threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, 
                                                          self.px_ref, 
                                                          focal=self.focal, 
                                                          pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.px_ref = self.px_cur

    def update(self, img):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and 
               img.shape[1]==self.cam.width), "Error"
        self.new_frame = img
        if(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame