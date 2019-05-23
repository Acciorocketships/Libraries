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

	# Calculates the camera matrix where all points in the region are defined after undistortion
	Kopt, _ = cv2.getOptimalNewCameraMatrix(K, d, imageSize=(img.shape[1], img.shape[0]), alpha=0, centerPrincipalPoint=False)

	# Undistort and apply change camera matrix to Kopt
	mapx, mapy = cv2.initUndistortRectifyMap(K, d, R=None, newCameraMatrix=Kopt, size=(img.shape[1], img.shape[0]), m1type=cv2.CV_16SC2)
	imgnew = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

	if returnK:
		# change Kopt to vertical coordinate first, horizontal coordinate second
		Kopt = swapK(Kopt)
		return imgnew, Kopt
	else:
		return imgnew


def swapK(K):
	# Swaps the order of x and y in the intrinsic matrix K
	Knew = K.copy()
	Knew[[0,1],:] = K[[1,0],:]
	Knew[[0,1],:] = Knew[[1,0],:]
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


def vanishingPoints(img, show=False):

	# Returns numVP vanishing points in the image via edge detection and ransac

	inlierThreshold = 3
	ransacIter = 1000
	smoothing = 2

	edgelets1 = compute_edgelets(img, sigma=smoothing)
	vp1 = ransac_vanishing_point(edgelets1, ransacIter, threshold_inlier=inlierThreshold) # Find first vanishing point
	vp1 = vp1 / vp1[2]
	if show:
		vis_edgelets(img, edgelets1)
		vis_model(img, vp1)

	edgelets2 = remove_inliers(vp1, edgelets1, 2*inlierThreshold) # Remove inlier to remove dominating direction.
	vp2 = ransac_vanishing_point(edgelets2, ransacIter, threshold_inlier=inlierThreshold) # Find second vanishing point
	vp2 = vp2 / vp2[2]
	if show:
		vis_edgelets(img, edgelets2)
		vis_model(img, vp2)

	vps = np.stack((vp1, vp2), axis=1)
	vps = np.stack((vps[1,:], vps[0,:]), axis=0)

	return vps










































## Helper Functions ##

def compute_edgelets(image, sigma=3):
	"""Create edgelets as in the paper.

	Uses canny edge detection and then finds (small) lines using probabilstic
	hough transform as edgelets.

	Parameters
	----------
	image: ndarray
		Image for which edgelets are to be computed.
	sigma: float
		Smoothing to be used for canny edge detection.

	Returns
	-------
	locations: ndarray of shape (n_edgelets, 2)
		Locations of each of the edgelets.
	directions: ndarray of shape (n_edgelets, 2)
		Direction of the edge (tangent) at each of the edgelet.
	strengths: ndarray of shape (n_edgelets,)
		Length of the line segments detected for the edgelet.
	"""
	gray_img = color.rgb2gray(image)
	edges = feature.canny(gray_img, sigma)
	lines = transform.probabilistic_hough_line(edges, line_length=3,
											   line_gap=2)

	locations = []
	directions = []
	strengths = []

	for p0, p1 in lines:
		p0, p1 = np.array(p0), np.array(p1)
		locations.append((p0 + p1) / 2)
		directions.append(p1 - p0)
		strengths.append(np.linalg.norm(p1 - p0))

	# convert to numpy arrays and normalize
	locations = np.array(locations)
	directions = np.array(directions)
	strengths = np.array(strengths)

	directions = np.array(directions) / \
		np.linalg.norm(directions, axis=1)[:, np.newaxis]

	return (locations, directions, strengths)


def edgelet_lines(edgelets):
	"""Compute lines in homogenous system for edglets.

	Parameters
	----------
	edgelets: tuple of ndarrays
		(locations, directions, strengths) as computed by `compute_edgelets`.

	Returns
	-------
	lines: ndarray of shape (n_edgelets, 3)
		Lines at each of edgelet locations in homogenous system.
	"""
	locations, directions, _ = edgelets
	normals = np.zeros_like(directions)
	normals[:, 0] = directions[:, 1]
	normals[:, 1] = -directions[:, 0]
	p = -np.sum(locations * normals, axis=1)
	lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
	return lines


def compute_votes(edgelets, model, threshold_inlier=5):
	"""Compute votes for each of the edgelet against a given vanishing point.

	Votes for edgelets which lie inside threshold are same as their strengths,
	otherwise zero.

	Parameters
	----------
	edgelets: tuple of ndarrays
		(locations, directions, strengths) as computed by `compute_edgelets`.
	model: ndarray of shape (3,)
		Vanishing point model in homogenous cordinate system.
	threshold_inlier: float
		Threshold to be used for computing inliers in degrees. Angle between
		edgelet direction and line connecting the  Vanishing point model and
		edgelet location is used to threshold.

	Returns
	-------
	votes: ndarry of shape (n_edgelets,)
		Votes towards vanishing point model for each of the edgelet.

	"""
	vp = model[:2] / model[2]

	locations, directions, strengths = edgelets

	est_directions = locations - vp
	dot_prod = np.sum(est_directions * directions, axis=1)
	abs_prod = np.linalg.norm(directions, axis=1) * \
		np.linalg.norm(est_directions, axis=1)
	abs_prod[abs_prod == 0] = 1e-5

	cosine_theta = dot_prod / abs_prod
	theta = np.arccos(np.abs(cosine_theta))

	theta_thresh = threshold_inlier * np.pi / 180
	return (theta < theta_thresh) * strengths


def remove_inliers(model, edgelets, threshold_inlier=10):
	"""Remove all inlier edglets of a given model.

	Parameters
	----------
	model: ndarry of shape (3,)
		Vanishing point model in homogenous coordinates which is to be
		reestimated.
	edgelets: tuple of ndarrays
		(locations, directions, strengths) as computed by `compute_edgelets`.
	threshold_inlier: float
		threshold to be used for finding inlier edgelets.

	Returns
	-------
	edgelets_new: tuple of ndarrays
		All Edgelets except those which are inliers to model.
	"""
	inliers = compute_votes(edgelets, model, 10) > 0
	locations, directions, strengths = edgelets
	locations = locations[~inliers]
	directions = directions[~inliers]
	strengths = strengths[~inliers]
	edgelets = (locations, directions, strengths)
	return edgelets


def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
	"""Estimate vanishing point using Ransac.

	Parameters
	----------
	edgelets: tuple of ndarrays
		(locations, directions, strengths) as computed by `compute_edgelets`.
	num_ransac_iter: int
		Number of iterations to run ransac.
	threshold_inlier: float
		threshold to be used for computing inliers in degrees.

	Returns
	-------
	best_model: ndarry of shape (3,)
		Best model for vanishing point estimated.
	"""
	locations, directions, strengths = edgelets
	lines = edgelet_lines(edgelets)

	num_pts = strengths.size

	arg_sort = np.argsort(-strengths)
	first_index_space = arg_sort[:num_pts // 5]
	second_index_space = arg_sort[:num_pts // 2]

	best_model = None
	best_votes = np.zeros(num_pts)

	for ransac_iter in range(num_ransac_iter):
		ind1 = np.random.choice(first_index_space)
		ind2 = np.random.choice(second_index_space)

		l1 = lines[ind1]
		l2 = lines[ind2]

		current_model = np.cross(l1, l2)

		if np.sum(current_model**2) < 1 or current_model[2] == 0:
			# reject degenerate candidates
			continue

		current_votes = compute_votes(
			edgelets, current_model, threshold_inlier)

		if current_votes.sum() > best_votes.sum():
			best_model = current_model
			best_votes = current_votes

	return best_model


def vis_edgelets(image, edgelets, show=True):
    """Helper function to visualize edgelets."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        plt.plot(xax, yax, 'r-')

    if show:
        plt.show()


def vis_model(image, model, show=True):
    """Helper function to visualize computed model."""
    import matplotlib.pyplot as plt
    edgelets = compute_edgelets(image)
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vis_edgelets(image, edgelets, False)
    vp = model / model[2]
    plt.plot(vp[0], vp[1], 'bo')
    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()



