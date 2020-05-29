import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

def projected_quaternion_initializer(shape, dtype=None):
    assert shape[1] == 4
    (num_rotations, _, quaternion_dim) = shape

    # http://planning.cs.uiuc.edu/node198.html
    (u1, u2, u3) = tf.random.uniform((3, num_rotations), maxval=1.)
    r = tf.sqrt(1 - u1)*tf.sin(2*math.pi*u2)
    x = tf.sqrt(1 - u1)*tf.cos(2*math.pi*u2)
    y = tf.sqrt(u1)*tf.sin(2*math.pi*u3)
    z = tf.sqrt(u1)*tf.cos(2*math.pi*u3)

    (a, b, c, d) = tf.random.uniform((4, num_rotations, quaternion_dim), 0., 1.)

    elements = []
    for (u, v) in zip((a, b, c, d), (r, x, y, z)):
        norm = tf.expand_dims(v, axis=-1)/tf.reduce_sum(u, axis=-1, keepdims=True)
        elements.append(u*norm)

    return tf.concat([tf.expand_dims(v, -2) for v in elements], axis=1)

@tf.function
def rotate(quat, vec):
    real = K.expand_dims(quat[..., 0], -1)
    imag = quat[..., 1:]
    result = (real**2 - K.sum(imag**2, axis=-1, keepdims=True))*vec
    result = result + 2*real*tf.linalg.cross(imag, vec)
    result = result + 2*K.sum(imag*vec, axis=-1, keepdims=True)*imag
    return result

class QuaternionRotation(keras.layers.Layer):
    def __init__(self, num_rotations, quaternion_dim=6, include_reverse=True, *args, **kwargs):
        self.num_rotations = num_rotations
        self.quaternion_dim = quaternion_dim
        self.include_reverse = include_reverse
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.quaternion_weight = self.add_weight(
            shape=(self.num_rotations, 4, self.quaternion_dim),
            initializer=projected_quaternion_initializer,
            name='pre_projected_quaternions'
        )

    @property
    def quaternions(self):
        quaternions = K.sum(self.quaternion_weight, axis=-1)
        quaternions = tf.linalg.normalize(quaternions, axis=-1)[0]
        return quaternions

    @property
    def training_quaternions(self):
        quaternions = self.quaternions

        if self.include_reverse:
            conj = quaternions*tf.constant([(-1., 1, 1, 1)])
            quaternions = tf.concat([quaternions, conj], axis=0)
        return quaternions

    def call(self, inputs):
        # (whatever, 3) -> (whatever, num_rotations, 3)
        replicated = K.expand_dims(inputs, -2)
        shape = K.int_shape(replicated)
        replicas = [1]*len(shape)
        replicas[-2] = self.num_rotations*(2 if self.include_reverse else 1)
        replicated = K.tile(replicated, replicas)

        # (num_rotations, 4, d) -> (num_rotations, 4)
        quaternions = self.training_quaternions
        # (num_rotations, 4) -> (whatever, num_rotations, 4)
        for _ in range(2, len(shape)):
            quaternions = K.expand_dims(quaternions, axis=-3)

        symbolic_shape = K.shape(replicated)
        replicas = [symbolic_shape[i] for i in range(len(shape) - 2)] + [1, 1]
        quaternions = K.tile(quaternions, replicas)

        return rotate(quaternions, replicated)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            num_rotations=self.num_rotations,
            quaternion_dim=self.quaternion_dim,
            include_reverse=self.include_reverse,
        ))
        return config

class QuaternionRotoinversion(QuaternionRotation):
    def call(self, inputs):
        return super().call(-inputs)

keras.utils.get_custom_objects().update(dict(
    QuaternionRotation=QuaternionRotation,
    QuaternionRotoinversion=QuaternionRotoinversion,
))
