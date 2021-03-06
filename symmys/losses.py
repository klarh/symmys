import numpy as np
import tensorflow.keras.backend as K

def mean_sqrt_rsq(pred, reference):
    """Returns mean(sqrt(R^2)) for a set of reference points."""
    expanded = K.expand_dims(pred, -2)

    delta = expanded - reference[np.newaxis]
    rsq = K.sum(delta**2, axis=-1)
    result = K.mean(K.sqrt(rsq), axis=-1)
    return K.mean(result, axis=-1)

def mean_exp_rsq(pred, reference, r_scale=1.):
    """Returns mean(1 - exp(-R^2/r_scale^2)) for a set of reference points."""
    expanded = K.expand_dims(pred, -2)

    delta = expanded - reference[np.newaxis]
    rsq = K.sum(delta**2, axis=-1)
    result = K.mean(1 - K.exp(-rsq/r_scale**2), axis=-1)
    return K.mean(result, axis=-1)
