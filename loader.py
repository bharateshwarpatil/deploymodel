import numpy as np
import os
import pickle
import torch
import torch.utils.data as data

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73




def load_data(id):
    dataPath = '/Users/bharat/Documents/work/ml/MRNet/vol05/'+id+'-5.pck'

    with open(dataPath, 'rb') as file_handler:  # Must use 'rb' as the data is binary
        vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    vol = vol[:, pad:-pad, pad:-pad]

    # standardize
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

    # normalize
    vol = (vol - MEAN) / STDDEV

    # convert to RGB
    vol = np.stack((vol,) * 3, axis=1)

    vol_tensor = torch.FloatTensor(vol)

    return vol_tensor


