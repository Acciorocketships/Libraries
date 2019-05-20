import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))
import cv2
import numpy as np
from imgstream import Stream


class colorDetect:

    def __init__(self):
        if hasros:
            # Initialize ROS Node
            rospy.init_node("hsv")
            # Detection Centroid Publisher (msg.x = x centroid, msg.y = y centroid, msg.theta = area)
            self.roi_pub = rospy.Publisher('dragonfly/roi', Pose2D, queue_size=10)
            # Estimated position
            self.pose_pub = rospy.Publisher("dragonfly/target_pose", PoseStamped, queue_size=1, latch=True)
            # Detection Covariance Publisher ([[msg.ixx, msg.ixy],[msg.ixy, msg.iyy]])
            self.roi_cov_pub = rospy.Publisher('dragonfly/roi_cov', Inertia, queue_size=10)
            # Computer Vision Mask Publisher
            self.image_pub = rospy.Publisher('dragonfly/mask', Image, queue_size=1)
            # Camera image subscriber
            self.simulation = rospy.get_param("~simulation", False)
            self.camera_name = rospy.get_param("~camera_name", "camera_usb")
            self.camera_suffix = rospy.get_param("~camera_suffix", "image_raw")
            self.filtered_position = None
            self.image_sub = rospy.Subscriber("/{}/{}".format(self.camera_name, self.camera_suffix), Image,
                                              self.callback)
            # TF Frame Lookup objects
            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30))  # type: tf2_ros.Buffer
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, )  # type: tf2_ros.TransformListener
            # Other
            self.bridge = CvBridge()

    def callback(self, data):
        # type: (Image) -> None
        try:
            T, R = self.get_traslation_rotation(data.header)
        except tf2_ros.LookupException:
            pass
        if hasros:
            # Convert image to cv2 format
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
        else:
            cv_image = data

        # Convert BGR to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image
        lower_mag = np.array([165, 70, 100])  # lower_mag=np.array([165,70,100])
        upper_mag = np.array([185, 180, 255])  # upper_mag=np.array([185,180,255])
        mask = cv2.inRange(hsv, lower_mag, upper_mag)
        mask = cv2.dilate(mask, None, iterations=1)

        # idxs = np.argwhere(mask>0)
        # if len(idxs) > 0:
        #     import random
        #     idx = idxs[random.randint(0,len(idxs)-1)]
        #     print("HSV:", hsv[idx[0],idx[1]])

        # Find centroids and area (and show them on mask)
        m = cv2.moments(mask)
        masked = Stream.mask(mask, cv_image, alpha=0.5)  # Overlay mask on image
        area = m['m00']
        # rospy.loginfo('area: ' + str(area))
        if area != 0:
            cx = m['m10'] / area
            cy = m['m01'] / area  # gets coords of mask
        else:
            if hasros:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(masked, "bgr8"))
                return
            else:
                return masked
        masked = Stream.mark(masked, (cx, cy, ((area / 3.14) ** 0.5 / 4)))
        # rospy.loginfo('cx: ' + str(cx) + "  cy: " + str(cy))

        # calculate real world coordinates
        K = self.get_intrinsic_matrix()
        height = None if self.simulation else 1080.0
        position = img2world((cx, cy), K, R, T, height=height)
        if self.filtered_position is None:
            self.filtered_position = position
        else:
            alpha = 0.5
            self.filtered_position[0] = position[0] * alpha + self.filtered_position[0] * (1-alpha)
            self.filtered_position[1] = position[1] * alpha + self.filtered_position[1] * (1-alpha)
        pose_stamped = to_pose_stamped(self.filtered_position, "map")
        pose_stamped.header.stamp = data.header.stamp
        self.pose_pub.publish(pose_stamped)

        # Calculate covariance
        u11 = (m['m11'] - cx * m['m01']) / m['m00']
        u20 = (m['m20'] - cx * m['m10']) / m['m00']
        u02 = (m['m02'] - cy * m['m01']) / m['m00']

        if hasros:

            # Publish Centroid and Area
            roi = Pose2D()
            roi.x = cx
            roi.y = cy
            roi.theta = area
            self.roi_pub.publish(roi)

            # Publish Covariance
            roi_cov = Inertia()
            roi_cov.ixx = u20
            roi_cov.ixy = u11
            roi_cov.iyy = u02
            self.roi_cov_pub.publish(roi_cov)

            # Show Image
            # if not rospy.get_param("~headless", False):
            #     Stream.show(masked, "Mask", shape=(720,1280))

            # Publish Image Mask
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(masked, "bgr8"))  # convert back to img message
            except CvBridgeError as e:
                print(e)
        else:
            print("x:", cx, ", y:", cy, ", area:", area)
            # cov = np.array([[u20,u11],[u11,u02]])
            # print("Cov:", cov)
            return masked

    def get_traslation_rotation(self, header):
        """
        Get a tuple of the translation and rotation matrices.
        :param header: The header of the image being processed.
        :type header: Header
        :return: Tuple of (Translation Vector, Rotation Matrix)
        :rtype: (np.ndarray(3,1), np.ndarray(3,3))
        """
        transform = self.tf_buffer.lookup_transform(self.camera_name + "_optical", "map",
                                                    rospy.Time(0), rospy.Duration(1))  # type: tf2_ros.TransformStamped

        translation = to_numpy(transform.transform.translation)
        q = transform.transform.rotation
        rotation_matrix = quaternion_matrix((q.x, q.y, q.z, q.w))[:3, :3]
        return (translation, rotation_matrix)

    def get_intrinsic_matrix(self):
        if self.simulation:
            return np.array([
                [585.756071, 0, 320.5],
                [0, 10, 240.5],
                [0, 0, 1]
            ])
        else:
            return np.array([
                [1586, 0, 557],
                [0, 1588, 916],
                [0, 0, 1]
            ])


def img2world(pos, K, R, T, plane=(0, 0, 1, 0), shape=None):
    # pos is a tuple of image coordinates. pos = (u, v) where x=image.shape[0]-u and y=v
    # R is a rotation from world coordinates to drone coordinates
    # T is a translation from world coordinates to drone coordinates
    # K is the camera intrinsics matrix [f 0 x0; 0 f y0; 0 0 1]
    # plane = [A B C D] is the plane Ax+By+Cz=D that we assume the point is on in world coordinates.
    # shape = (imgheight, imgwidth). Specify this to convert from [u v 1] to [x y 1], where u: down, v: right, origin top left
    # If shape==None, then it assumes coordinates are already [x y 1], where x: up, y: right, origin bottom left

    # [u v 1] = K * (R * plane * [Xw Yw 1] + T)
    # [plane]^(-1) * R' * (K^(-1) * [u v 1] - T) = [Xw Yw 1]

    T = T.reshape((3,1))

    # Compose [u v 1] image coordinates
    pos = (pos[0], pos[1])  # invert x (vertical) axis
    pImg = np.array([[pos[0]], [pos[1]], [1]])

    # Image to body coordinates [Xb, Yb, Zb]
    pBody = np.dot(np.linalg.inv(K), pImg)

    # Apply R transformation giving the coordinates in the system with the transformation of the body but aligned with the world
    pBodyAligned = np.dot(R, pBody)
    # Get scale so ray intersects with desired plane (scale is the length of the ray)
    num = plane[3] - np.dot(T.T, plane[:3])
    denom = np.dot(pBodyAligned.T, plane[:3])
    if denom != 0:
        scale = num / denom
    else:
        scale = 0

    # Find point in world coordinates
    pWorld = scale * pBodyAligned + T

    return pWorld


def main():
    detector = colorDetect()
    if hasros:
        rospy.spin()
    else:
        stream = Stream(mode='cam', src='1')
        for img in stream:
            mask = detector.callback(img)
            # Stream.show(img, "Img")
            Stream.show(mask, "Mask", shape=(720, 1280))


def test_img2world():
    # Generate K matrix
    res = [1080,1920]
    f = 1000
    x0 = res[0] / 2
    y0 = res[1] / 2
    K = np.zeros((3,3))
    K[0,0] = f
    K[1,1] = f
    K[2,2] = 1
    K[0,2] = x0
    K[1,2] = y0
    # Generate Rotation
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('zyx', [10,-10,20], degrees=True)
    Rcw = rot.as_dcm() # Rotation of Camera wrt World
    # Generate Translation
    Tcw = np.array([0.5,-0.3,-4]) # Translation of Camera wrt World
    # Generate Point
    Pw = np.array([0.1,0.2,0])
    # Calculate Pixel
    Pc = K @ (Rcw.T @ Pw + -Rcw.T@Tcw)
    Pc = Pc / Pc[2]
    # Test
    Ppred = img2world(Pc,K=K,R=Rcw,T=Tcw,plane=[0,0,1,0])
    print("K", K)
    print("\n")
    print("R", Rcw)
    print("\n")
    print("T", Tcw)
    print("\n")
    print("Pc:", Pc)
    print("\n")
    print("Pw:", Pw)
    print("\n")
    print("Ppred:", Ppred)


if __name__ == '__main__':
    #main()
    test_img2world()

# mavros = autopilot.Mavros()
# quad = core.Multirotor(mavros, frequency=10)
# quad.takeoff()
# quad.set_position([0, 0, 10], blocking=True)
# rospy.sleep(3)
# quad.land()


# Guassian Blur if needed
# cv_image = cv2.GaussianBlur(cv_image,(3,3),0)


# Determine if it is a rectangle
# try:
#     _, cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts:
#         approx = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True)
#         cv2.drawContours(cv_image, [approx], -1, (0,255,0), 4)
#         if len(approx)==4:
#             print "rectangle"
#         else:
#             print "not"
# except:
#     print("end")


# https://github.com/pkrish2/Tracking-Drone/blob/pranab/takeoff.py
