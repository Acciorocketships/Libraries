import cv2
import numpy as np
from skimage import feature, color, transform


def undistort(img, K, d, returnK=False):
	# img: nxmx3 image to be undistorted
	# K: intrinsic matrix of the camera that took img
	# d: [k1,k2] radial distortion coefficients of the camera that took img
	# returnK: If false, function returns newimg. If true, function returns (newimg,newK)

	# distortion P1, P2 are 0
	d = np.array([d[0], d[1], 0, 0])

	# opencv uses horizontal as first coordinate and vertical as second, so change K and size accordingly
	size = (img.shape[1], img.shape[0])
	K = swapK(K)
	K[1,2] = img.shape[0] - K[1,2]

	# Calculates the camera matrix where all points in the region are defined after undistortion
	Kopt, _ = cv2.getOptimalNewCameraMatrix(K, d, imageSize=size, alpha=0, centerPrincipalPoint=False)

	# Undistort and apply change camera matrix to Kopt
	mapx, mapy = cv2.initUndistortRectifyMap(K, d, R=None, newCameraMatrix=Kopt, size=size, m1type=cv2.CV_16SC2)
	imgnew = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

	if returnK:
		# change Kopt to vertical coordinate first, horizontal coordinate second
		Kopt = swapK(Kopt)
		Kopt[0,2] = img.shape[0] - Kopt[0,2]
		return imgnew, Kopt
	else:
		return imgnew


def swapK(K):
	# Swaps the order of x and y in the intrinsic matrix K
	Knew = K.copy()
	Knew[[0,1],:] = K[[1,0],:]
	Knew[:,[0,1]] = Knew[:,[1,0]]
	return Knew


def pixToWorldPlane(pos, K, plane=np.array([0, 0, 1, 0]), R=np.eye(3), T=np.zeros((3,1)), shape=None):
	# Finds the 3D world position given pixel coords, assuming the object is on a known plane

	# pos is a tuple of image coordinates
	# R is the 3x3 rotation of the camera relative to the world
	# T is the 3x1 translation of the camera relative to the world
	# If R=I and T=[0,0,0] and the plane is given in body coordinates, then the returned 3D position is given in body coordinates
	# K is the camera intrinsics matrix [f 0 x0; 0 f y0; 0 0 1]
	# plane = [A B C D] is the plane Ax+By+Cz=D that we assume the point is on (in world coordinates).
	# shape = (imgheight, imgwidth). Specify this to convert from [u v 1] to [x y 1], where u: down, v: right, origin top left
	# If shape==None, then it assumes pos is already [x y 1], where x: up, y: right, origin bottom left

	# Pc = K^(-1) * [x y 1]
	# lambda * Pc = R' * Pw + (-R'T)
	# Pw = (lambda * R * Pc) + T
	# [A B C] • (lambda * R * Pc + T) = Pw
	# lambda = (D - [A B C] • T) / ([A B C] • (R * Pc))

	# Put inputs into the proper format
	pos = np.array(pos)
	K = np.array(K)
	R = np.array(R)
	T = np.array(T)
	plane = np.array(plane)
	T = T.reshape((3,1))

	# Compose [x y 1] image coordinates
	if shape is not None:
		pos = np.array([shape[0]-pos[0], pos[1]])  # invert vertical axis to go from [u v 1] to [x y 1]
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
		scale = float("inf")

	# Find point in world coordinates
	pWorld = scale * pBodyAligned
	pWorld = pWorld.reshape((3,))

	return pWorld


def pixToWorldDist(pos1, pos2, dist, K, plane=np.array([0, 0, 1]), R=np.eye(3), T=np.zeros((3,1)), shape=None, returnD=False):
	# Find the 3D world position of two points given their pixel coordinates,
	# the distance between them in the world, and the normal of the plane they are on (offset not needed)

	# pos1, pos2 are arrays/tuples of [u v] (if shape is given) or [x y] (if shape is None) of image coordinates
	# dist is the displacement in meters between the 3D points corresponding to pos1 and pos2 in the real world
	# R is the 3x3 rotation of the camera relative to the world
	# T is the 3x1 translation of the camera relative to the world
	# If R=I and T=[0,0,0] and the plane is given in body coordinates, then the returned 3D position is given in body coordinates
	# K is the camera intrinsics matrix [f 0 x0; 0 f y0; 0 0 1]
	# plane = [A B C D] is the plane Ax+By+Cz=D that we assume the point is on (in world coordinates).
	# shape = (imgheight, imgwidth). Specify this to convert from [u v 1] to [x y 1], where u: down, v: right, origin top left
	# If shape==None, then it assumes pos is already [x y 1], where x: up, y: right, origin bottom left
	# returnD is a bool that indicates whether or not to return the calculated D for the plane Ax+By+Cz=D that contains the points

	# Returns (P1, P2), the 3D points in world coordinates corresponding to 

	# Put inputs into the proper format
	pos1 = np.array(pos1)
	pos2 = np.array(pos2)
	K = np.array(K)
	R = np.array(R)
	T = np.array(T)
	plane = np.array(plane)
	T = T.reshape((3,1))

	# Compose [x y 1] image coordinates
	if shape is not None:
		pos1 = np.array([shape[0]-pos1[0], pos1[1]])  # invert vertical axis to go from [u v 1] to [x y 1]
		pos2 = np.array([shape[0]-pos2[0], pos2[1]])
	pImg1 = np.array([[pos1[0]], [pos1[1]], [1]])
	pImg2 = np.array([[pos2[0]], [pos2[1]], [1]])

	# Image to body coordinates [Xb, Yb, Zb]
	pBody1 = np.dot(np.linalg.inv(K), pImg1)
	pBody2 = np.dot(np.linalg.inv(K), pImg2)

	# Apply R transformation giving the coordinates in the system with the transformation of the body but aligned with the world
	pBodyAligned1 = np.dot(R, pBody1)
	pBodyAligned2 = np.dot(R, pBody2)

	# Find the distance to plane intersection for D=0 in Ax+By+Cz=D
	try:
		lambda1_test = -np.dot(T.T, plane[:3]) / np.dot(pBodyAligned1.T, plane[:3])
		lambda2_test = -np.dot(T.T, plane[:3]) / np.dot(pBodyAligned2.T, plane[:3])
	except ZeroDivisionError:
		raise ZeroDivisionError("One of the input points is parallel with the given plane")

	# Find corresponding distance between world coordinates for D=0
	pWorld1_test = lambda1_test * pBodyAligned1 + T
	pWorld2_test = lambda2_test * pBodyAligned2 + T
	dist_test = np.linalg.norm(pWorld1_test-pWorld2_test)

	# Find actual distances to plane intersection
	lambda1 = lambda1_test * (dist / dist_test)
	lambda2 = lambda2_test * (dist / dist_test)

	pWorld1 = lambda1 * pBodyAligned1 + T
	pWorld2 = lambda2 * pBodyAligned2 + T

	pWorld1 = pWorld1.reshape((3,))
	pWorld2 = pWorld2.reshape((3,))

	if returnD:
		 # Calculate D
		D = lambda1 * np.dot(pBodyAligned1.T, plane[:3]) + np.dot(T.T, plane[:3])
		return (pWorld1, pWorld2, D)
	else:
		return (pWorld1, pWorld2)


def PnP(imgpts, objpts, K=None, shape=None, ransac=False):

	# Input shapes -- imgpts: nx2, objpts: nx3, K: 3x3
	# If K or shape is None, then imgpts is in camera coordinates (meters not pixels). No x/y flipping or K matrix is applied.

	if type(imgpts)==list or type(imgpts)==tuple:
		imgpts = np.stack(imgpts, axis=0)
	if type(objpts)==list or type(objpts)==tuple:
		objpts = np.stack(objpts, axis=0)

	imgpts = imgpts.astype(np.float64)
	objpts = objpts.astype(np.float64)

	Rcoord = np.array([[0,-1,0],[1,0,0],[0,0,1]]) # opencv coordinate from wrt my coordinate frame

	if K is None or shape is None:
		K = np.eye(3)
		imgpts = np.stack((imgpts[:,0],imgpts[:,1]), axis=1) # if imgpts is nx3, make it nx2
	else:
		K = swapK(K) # convert to [horizontal, vertical] for opencv
		K[1,2] = shape[0] - K[1,2] # opencv imgcenter is relative to top left instead of bottom left
		imgpts = np.stack((imgpts[:,1],imgpts[:,0]), axis=1) # convert to [horizontal, vertical] for opencv
		objpts = (Rcoord.T @ objpts.T).T
	
	imgpts = np.expand_dims(imgpts, axis=1) # add extra dimension so solvpnp wont error out (now nx1x2)

	if ransac:
		_, rvec, T, inliers = cv2.solvePnPRansac(objpts, imgpts, K, np.array([[0,0,0,0]]), flags=cv2.SOLVEPNP_EPNP)
		print(inliers)
	else:
		_, rvec, T = cv2.solvePnP(objpts, imgpts, K, np.array([[0,0,0,0]]), flags=cv2.SOLVEPNP_EPNP)

	R = cv2.Rodrigues(rvec)[0] # axis angle to rotation matrix

	if not (K is None or shape is None):
		T = Rcoord @ T # transform back to unrotated coordinate frame (pre multiply for local transform)

	if ransac:
		return (R, T, inliers)
	else:
		return (R, T)




