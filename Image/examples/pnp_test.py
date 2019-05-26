import os,sys
sys.path.append(os.path.dirname(os.path.realpath("")))

from image import *
from imgstream import *


def test_pnp():
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
    rot = Rotation.from_euler('zyx', [10,0,0], degrees=True)
    Rcw = rot.as_dcm() # Rotation of Camera wrt World
    # Generate Translation
    Tcw = np.array([0.5,-0.3,-4]) # Translation of Camera wrt World
    # Generate Point
    Pw = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    # Calculate Pixel
    Pc = (( Rcw.T @ Pw.T + np.reshape(-Rcw.T@Tcw, (3,1)) )).T
    Pc = Pc / np.reshape(Pc[:,2], (4,1))
    # Solve PnP
    R, T = PnP(Pc, Pw, ransac=False)
    # Convert to (obj wrt cam) to (cam wrt obj) to compare to Rcw, Tcw
    T = -R.T @ T
    R = R.T
    # Validate
    PwPred = R @ (np.linalg.inv(K) @ Pc.T) + T
    # Print
    print("Pc:", Pc)
    print("\n")
    print("Pw:", Pw)
    print("\n")
    print("PwPred:", PwPred)
    print("\n")
    print("R:", R)
    print("\n")
    print("T:", T)
    print("\n")
    print("Rcw:", Rcw)
    print("\n")
    print("Tcw:", Tcw)



def test2_pnp():
    # Generate K matrix
    res = [1080,1920]
    f = 1000
    x0 = res[0] / 2 - 20
    y0 = res[1] / 2 - 20
    K = np.zeros((3,3))
    K[0,0] = f
    K[1,1] = f
    K[2,2] = 1
    K[0,2] = x0
    K[1,2] = y0
    # Generate Rotation
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('zyx', [10,0,0], degrees=True)
    Rcw = rot.as_dcm() # Rotation of Camera wrt World
    # Generate Translation
    Tcw = np.array([0.5,-0.3,-4]) # Translation of Camera wrt World
    # Generate Point
    Pw = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    # Calculate Pixel
    Pc = (K @ ( Rcw.T @ Pw.T + np.reshape(-Rcw.T@Tcw, (3,1)) )).T
    Pc = Pc / np.reshape(Pc[:,2], (-1,1))
    Pc[:,0] = res[0] - Pc[:,0] # convert origin to top left instead of bottom left
    # Solve PnP
    R, T = PnP(Pc, Pw, K, shape=res, ransac=True)
    # Convert to (obj wrt cam) to (cam wrt obj) to compare to Rcw, Tcw
    T = -R.T @ T
    R = R.T
    # Validate
    PwPred = R @ (np.linalg.inv(K) @ Pc.T) + T
    # Print
    print("Pc:", Pc)
    print("\n")
    print("Pw:", Pw)
    print("\n")
    print("PwPred:", PwPred)
    print("\n")
    print("R:", R)
    print("\n")
    print("T:", T)
    print("\n")
    print("Rcw:", Rcw)
    print("\n")
    print("Tcw:", Tcw)



if __name__ == '__main__':
    test2_pnp()



    