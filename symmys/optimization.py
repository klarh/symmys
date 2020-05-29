import collections
import functools

import numpy as np
import sklearn.cluster
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from .layers import QuaternionRotation, QuaternionRotoinversion
from .losses import mean_exp_rsq
from .utils import hash_sample

class PointRotations:
    """Finds rotations that leave a point cloud unchanged up to a permutation.

    This method optimizes a set of unit quaternions to match the
    distribution of transformed points to the set of unrotated
    points. Quaternions are then clustered by their axis of rotation
    and merged into N-fold rotation symmetries.

    :param num_rotations: Number of plain rotations (and rotoinversions, if enabled) to consider
    :param quaternion_dim: Optimizer dimension for quaternions (higher may make optimization easier at the cost of more expensive optimization steps)
    :param include_inversions: If True, include rotoinversions as well as rotations
    :param loss: Loss function to use; see :py:mod:`symmys.losses`
    """
    def __init__(self, num_rotations, quaternion_dim=8, include_inversions=True,
                 loss=mean_exp_rsq):
        self.num_rotations = num_rotations
        self.quaternion_dim = quaternion_dim
        self.include_inversions = include_inversions
        self.loss = loss

        self._model_dict = None

    @property
    def model(self):
        """Return the tensorflow model that will perform rotations."""
        if self._model_dict is None:
            self._model_dict = self.build_model()
        return self._model_dict['model']

    @property
    def rotation_layer(self):
        """Return the tensorflow.keras layer for rotations."""
        if self._model_dict is None:
            self._model_dict = self.build_model()
        return self._model_dict['rotation_layer']

    @property
    def rotoinversion_layer(self):
        """Return the tensorflow.keras layer for rotoinversions."""
        if self._model_dict is None:
            self._model_dict = self.build_model()
        return self._model_dict['rotoinversion_layer']

    def build_model(self):
        """Create the tensorflow model.

        This method can be replaced by child classes to experiment
        with different network architectures. The returned result
        should be a dictionary containing at least:

        - `model`: a `tensorflow.keras.models.Model` instance that replicates a given set of input points
        - `rotation_layer`: a layer with a `quaternions` attribute to be read
        - `rotoinversion_layer` (if inversions are enabled): a layer with a `quaternions` attribute to be read
        """
        result = {}

        inp = last = keras.layers.Input(shape=(3,))
        result['rotation_layer'] = rot_layer = QuaternionRotation(
            self.num_rotations, quaternion_dim=self.quaternion_dim)

        if self.include_inversions:
            result['rotoinversion_layer'] = conj_rot_layer = QuaternionRotoinversion(
                self.num_rotations, quaternion_dim=self.quaternion_dim)
            last = tf.concat([rot_layer(last), conj_rot_layer(last)], 1)
        else:
            last = rot_layer(last)

        result['model'] = keras.models.Model(inp, last)
        return result

    def fit(self, points, epochs=1024, early_stopping_steps=16,
            validation_split=.3, hash_sample_N=128,
            reference_fraction=.1, optimizer='adam', batch_size=256,
            valid_symmetries=12, extra_callbacks=[]):
        """Fit rotation quaternions and analyze the collective symmetries of a set of input points.

        This method builds a rotation model, fits it to the given
        data, and groups the found quaternions by their axis and
        rotation angle.

        After fitting, a map of symmetries will be returned: a
        dictionary of {N-fold: [axes]} containing all the axes about
        which each observed symmetry were found.

        :param points: Input points to analyze:: (N, 3) numpy array-like sequence
        :param epochs: Maximum number of epochs to train
        :param early_stopping_steps: Patience (in epochs) for early stopping criterion; training halts when the validation set loss does not improve for this many epochs
        :param validation_split: Fraction of training data to use for calculating validation loss
        :param hash_sample_N: Minimum number of points to use as reference data for the loss function (see :py:func:`hash_sample`)
        :param reference_fraction: Fraction of given input data to be hashed to form the reference data
        :param optimizer: Tensorflow/keras optimizer name or instance
        :param batch_size: Batch size for optimization
        :param valid_symmetries: Maximum degree of symmetry (N) that will be considered when identifying N-fold rotations
        :param extra_callbacks: Additional tensorflow callbacks to use during optimization
        """
        points = np.asarray(points)
        N = len(points)

        reference_N = int(reference_fraction*N)
        reference, train = points[:reference_N], points[reference_N:]
        reference = hash_sample(reference, hash_sample_N)

        reduce_lr_patience = int(early_stopping_steps/3.)
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=early_stopping_steps, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(
                patience=reduce_lr_patience, monitor='val_loss', factor=.5, verbose=False),
        ] + extra_callbacks

        try:
            import tensorflow_addons as tfa
            callbacks.append(tfa.callbacks.TQDMProgressBar(
                show_epoch_progress=False, update_per_second=1))
        except ImportError:
            pass

        model = self.model
        loss = self.loss(model.output, reference)
        model.add_loss(loss)
        model.compile(optimizer, loss=None)
        model.fit(
            train, train, validation_split=validation_split, verbose=False,
            batch_size=batch_size, callbacks=callbacks, epochs=epochs)

        self.history = model.history.history

        if isinstance(valid_symmetries, int):
            valid_symmetries = range(1, valid_symmetries + 1)
        Ns = np.array(list(sorted(valid_symmetries)))

        symmetries = collections.defaultdict(list)
        for (symmetry, axis) in zip(*self._cluster_quaternions(
                self.rotation_layer.quaternions.numpy(), Ns)):
            if symmetry == 1:
                continue
            symmetries[symmetry].append(axis)

        if self.include_inversions:
            for (symmetry, axis) in zip(*self._cluster_quaternions(
                    self.rotoinversion_layer.quaternions.numpy(), Ns)):
                symmetries[-symmetry].append(axis)

        self.symmetries = dict(symmetries)

        return self.symmetries

    @staticmethod
    def _cluster_quaternions(quats, Ns, tolerance=.0125):
        axes = quats[:, 1:].copy()
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        filt = np.logical_and(np.isfinite(quats[:, 0]), np.all(np.isfinite(axes), axis=-1))
        quats, axes = quats[filt], axes[filt]

        distances = 1 - np.abs(np.sum(axes[:, np.newaxis]*axes[np.newaxis], axis=-1))
        distances = np.clip(distances, 0, 1)

        dbscan = sklearn.cluster.DBSCAN(eps=.0125, min_samples=1, metric='precomputed')
        cluster_indices = dbscan.fit_predict(distances)

        clusters = collections.defaultdict(list)
        for (i, theta, axis) in zip(cluster_indices, 2*np.arccos(quats[:, 0]), axes):
            clusters[i].append((theta, axis))

        averaged_axes = []
        symmetries = []
        for i, axisangles in clusters.items():
            ref_axis = axisangles[0][1]
            ref_axis *= np.sign(ref_axis[2])

            thetas = [theta for (theta, _) in axisangles]
            axis = np.mean([
                axis*np.sign(np.dot(axis, ref_axis)) for (_, axis) in axisangles], axis=0)
            axis /= np.linalg.norm(axis, keepdims=True)
            if np.any(np.logical_not(np.isfinite(axis))):
                axis[:] = (0, 0, 1)

            thetas = np.array(thetas)

            psis = np.exp(thetas*Ns[:, np.newaxis]*1j)
            psis /= Ns[:, np.newaxis]
            psis = 0.5*np.mean(psis + 1, axis=-1)
            psis = np.abs(psis) - .5
            symmetry = np.argmax(psis) + 1
            factors = 2*psis*Ns

            averaged_axes.append(axis)
            symmetries.append(symmetry)

        return symmetries, averaged_axes
