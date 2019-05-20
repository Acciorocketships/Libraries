import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *


def test_pixToWorldDist():
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
    Pw1 = np.array([0.1,0.2,0])
    Pw2 = np.array([-0.3,0.1,0])
    # Calculate Pixel
    Pc1 = K @ (Rcw.T @ Pw1 + -Rcw.T@Tcw)
    Pc2 = K @ (Rcw.T @ Pw2 + -Rcw.T@Tcw)
    Pc1 = Pc1 / Pc1[2]
    Pc2 = Pc2 / Pc2[2]
    dist = np.linalg.norm(Pw1 - Pw2)
    # Test
    Pw1pred, Pw2pred, D = pixToWorldDist(Pc1, Pc2, dist, K=K, R=Rcw, T=Tcw, plane=[0,0,1], returnD=True)
    print("K", K)
    print("\n")
    print("R", Rcw)
    print("\n")
    print("T", Tcw)
    print("\n")
    print("Pc1:", Pc1, "Pc2:", Pc2)
    print("\n")
    print("dist:", dist)
    print("\n")
    print("Pw1:", Pw1, "Pw2:", Pw2)
    print("\n")
    print("Pw1pred:", Pw1pred, "Pw2pred:", Pw2pred)



if __name__ == '__main__':
    test_pixToWorldDist()



    