import numpy as np
import pickle
from easydict import EasyDict as edict
from rpy2 import robjects as robj


# some utils 
def array2d2Robj(mat):
    """
    Converts a 2D numpy array to an R matrix object.

    Args:
        mat (numpy.ndarray): A 2D numpy array.

    Returns:
        r.matrix: An R matrix object.
    """
    mat_vec = mat.reshape(-1)
    mat_vecR = robj.FloatVector(mat_vec)
    matR = robj.r.matrix(mat_vecR, nrow=mat.shape[0], ncol=mat.shape[1], byrow=True)
    return matR
def get_ball_cor(x, y):
    """
    Calculates the ball correlation coefficient between two arrays x and y.

    Parameters:
    x (numpy.ndarray): A 1-D or 2-D array.
    y (numpy.ndarray): A 1-D array with the same number of samples as x.

    Returns:
    float: The ball correlation coefficient between x and y.
    """
    r = robj.r
    r["library"]("Ball");
    assert y.shape[0] == x.shape[0], "There should be the same number of samples"
    assert x.ndim <= 2, "x is at most 2-d array"
    assert y.ndim <= 2, "y is at most 2-d array"
    if y.ndim == 1:
        yR = robj.FloatVector(y);
    elif x.ndim == 2:
        yR = array2d2Robj(y)
    if x.ndim == 1:
        xR = robj.FloatVector(x);
    elif x.ndim == 2:
        xR = array2d2Robj(x)
    bcor_v = np.array(r['bcor'](yR, xR))[0]
    return bcor_v

def load_pkl_folder2dict(folder, excluding=[], including=["*"], verbose=True):
    """The function is to load pkl file in folder as an edict
        args:
            folder: the target folder
            excluding: The files excluded from loading
            including: The files included for loading
            Note that excluding override including
    """
    if not isinstance(including, list):
        including = [including]
    if not isinstance(excluding, list):
        excluding = [excluding]
        
    if len(including) == 0:
        inc_fs = []
    else:
        inc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in including])))
    if len(excluding) == 0:
        exc_fs = []
    else:
        exc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in excluding])))
    load_fs = np.setdiff1d(inc_fs, exc_fs)
    res = edict()
    for fil in load_fs:
        res[fil.stem] = load_pkl(fil, verbose)                                                                                                                                  
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False, verbose=True):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force, verbose=verbose)

# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False, verbose=True):
    if not fil.parent.exists():
        fil.parent.mkdir()
        if verbose:
            print(fil.parent)
            print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        if verbose:
            print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        if verbose:
            print(f"{fil} exists! Use is_force=True to save it anyway")
        else:
            pass
