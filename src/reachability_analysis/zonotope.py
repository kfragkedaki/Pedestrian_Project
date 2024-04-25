import pypolycontain as pp
import numpy as np


def zonotope(c_z: np.ndarray, G_z: np.ndarray) -> pp.zonotope:
    """ Zonotope creation 

        Parameters:
        -----------
        c_z : np.ndarray
            The center of the measurement
        G_z : np.ndarray    
            Generator matrix for zonotope
    """
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z.reshape(c_z.shape[0], 1), G=G_z.reshape(G_z.shape[0], -1))
    
def matrix_zonotope(C_M: np.ndarray, G_M: np.ndarray) -> pp.zonotope:
    """ Zonotope creation 

        Parameters:
        -----------
        c_z : np.ndarray
            The center of the measurement
        G_z : np.ndarray    
            Generator matrix for zonotope
    """
    assert C_M.shape[0] == G_M.shape[1]
    return pp.zonotope(x=C_M, G=G_M)
