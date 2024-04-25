import numpy as np
import pickle
import os


ROOT = os.getcwd()

def load_data(file: str = "sind.pkl") -> np.ndarray:
    """Load previously pickled data

    Parameters:
    -----------
    file : str (default = 'sind.pkl')
        File-name of the pickled file.
    """
    _f = open(ROOT + "/resources/" + file, "rb")
    return pickle.load(_f)
