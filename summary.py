import tensorflow as tf


def get_mean_grad(grads):
    mean_abs = [tf.reduce_mean(tf.abs(_)) for _ in grads]
    mean_abs = tf.reduce_mean(tf.stack(mean_abs))
    return mean_abs

