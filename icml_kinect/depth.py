import OpenEXR
import Imath
import array
import numpy as np

import bodyparts as bodyparts

# Load depth data from an exr file
def load_depth( filename ):
    fileH = OpenEXR.InputFile(filename)

    # Compute the size
    dw = fileH.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B, Z) = [array.array('f', fileH.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "Z") ]

    depth = np.fromstring(fileH.channel("Z", FLOAT), dtype = np.float32)
    depth.shape = (size[1], size[0]) # Numpy arrays are (row, col)

    return depth

# Reconstruct depth image
def reconstructDepthImage(depth, labels):
    (M, N) = depth.shape
    rgb = 255.0 * np.ones((M,N,3))
    for m in range(M):
        for n in range(N):
            rgb[m,n,0] = (depth[m,n]-1.0)/5.0
            rgb[m,n,1] = (depth[m,n]-1.0)/5.0
            rgb[m,n,2] = (depth[m,n]-1.0)/5.0
    return rgb