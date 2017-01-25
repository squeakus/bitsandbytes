# Topology optimisation code adapted from matlab
# U = Displacement
# F = Force Vector
# K = Global stiffness matrix

import numpy as np
import scipy.sparse as sps
from mlabwrap import mlab


def optimise(width, height, volfrac, penalty, filtersize):
    loop = 0
    change = 1.0
    area = np.empty((height,width)) #height equals row, width = columns
    area.fill(volfrac)

    displacement = mlab.finelem(width, height, area, penalty)

            
def stiffnessmatrix():
    E = 1.0 #youngs modulus
    nu = 0.3 #poisson ratio
    #stiffness vector
    k = [ (1./2)-(nu/6), 1./8+nu/8, -1./4-nu/12, -1./8+3*nu/8, 
          -1./4+nu/12, -1./8-nu/8, nu/6, 1./8-3*nu/8];
    klist = [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
             [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
             [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
             [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
             [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
             [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
             [k[2], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
             [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]

    
    KE = np.dot(E/(1.-nu*nu), klist)
    np.set_printoptions(precision=4)
    print KE
    return KE

if __name__=='__main__':
    optimise(60, 20, 0.5, 3.0, 1.5)
