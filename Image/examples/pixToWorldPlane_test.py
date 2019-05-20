import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *


def test_pixToWorldPlane():
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
    Ppred = pixToWorldPlane(Pc,K=K,R=Rcw,T=Tcw,plane=[0,0,1,0])
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



def detect(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image
    lower_mag1 = np.array([0, 130, 160])  # lower_mag=np.array([165,70,100])
    upper_mag1 = np.array([10, 185, 190])  # upper_mag=np.array([185,180,255])
    lower_mag2 = np.array([170, 130, 160])  # lower_mag=np.array([165,70,100])
    upper_mag2 = np.array([180, 185, 190])  # upper_mag=np.array([185,180,255])
    mask1 = cv2.inRange(hsv, lower_mag1, upper_mag1)
    mask2 = cv2.inRange(hsv, lower_mag2, upper_mag2)
    mask = mask1 | mask2
    mask = cv2.dilate(mask, None, iterations=1)

    # Find centroids and area (and show them on mask)
    m = cv2.moments(mask)
    masked = Stream.mask(mask, img, alpha=0.5)  # Overlay mask on image
    area = m['m00']
    c_vert = 0
    c_horiz = 0
    if area != 0:
        c_vert = m['m01'] / area
        c_horiz = m['m10'] / area
    return np.array([c_vert,c_horiz]), masked



def test_pixToWorldPlane_live():

    K = np.array([[981.7,      0, 335.7],
              [    0, 1043.7, 636.4],
              [    0,      0,     1]])
    d = np.array([0.0977, 0.0012])

    stream = Stream(mode='cam', src=0)
    for img in stream:
        img, Knew = undistort(img, K, d, returnK=True)
        pc, masked = detect(img)
        P = pixToWorldPlane(pc, Knew, plane=[0,0,1,1], shape=img.shape)
        masked = Stream.mark(masked, str(P), size=1)
        masked = Stream.mark(masked, pc, size=2)
        Stream.show(masked, "Detection", shape=(720,1280))



if __name__ == '__main__':
    #test_pixToWorldPlane()
    test_pixToWorldPlane_live()



    