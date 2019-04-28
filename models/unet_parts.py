import math
import logging
import tensorflow as tf

class double_conv(tf.keras.Model):
    def __init__(self,  out_ch, normaliz=True, activ=True):
        super(double_conv, self).__init__()
        ops = []
        ops += [tf.keras.layers.Conv2D(out_ch, 3, padding='same')]
        if normaliz:
            ops += [tf.keras.layers.BatchNormalization()]
        if activ:
            ops += [tf.keras.layers.Activation('relu')]
        ops += [tf.keras.layers.Conv2D(out_ch, 3, padding='same')]
        if normaliz:
            ops += [tf.keras.layers.BatchNormalization()]
        if activ:
            ops += [tf.keras.layers.Activation('relu')]

        self.conv = tf.keras.models.Sequential(ops)

    def call(self, x):
        x = self.conv(x)
        return x


class inconv(tf.keras.Model):
    def __init__(self,  out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(out_ch)

    def call(self, x):
        out = self.conv(x)
        logging.debug('inconv: {}->{}'.format(x.shape[1:], out.shape[1:]))
        return out


class down(tf.keras.Model):
    def __init__(self,  out_ch, normaliz=True):
        super(down, self).__init__()
        self.mpconv = tf.keras.models.Sequential(
            [tf.keras.layers.MaxPool2D(2),
             double_conv(out_ch, normaliz=normaliz)
            ]
        )

    def call(self, x):
        out = self.mpconv(x)
        logging.debug('Downsample: {}->{}'.format(x.shape[1:], out.shape[1:]))
        return out


def up_bilinear(images):
    x_shape = images.get_shape().as_list()
    h, w = x_shape[1], x_shape[2]
    return tf.image.resize(images, size=(2*h, 2*w), method=tf.image.ResizeMethod.BILINEAR)
    
class up(tf.keras.Model):
    def __init__(self,  out_ch, normaliz=True, activ=True):
        super(up, self).__init__()
        self.up = tf.keras.layers.Lambda(up_bilinear)
        self.conv = double_conv( out_ch,
                                normaliz=normaliz, activ=activ)

    def call(self, x1, x2):
        x1 = self.up(x1)
        x = tf.concat([x2, x1], axis=-1)
        out = self.conv(x)
        logging.debug('Upsample: {}->{}'.format(x.shape[1:], out.shape[1:]))
        return out


class outconv(tf.keras.Model):
    def __init__(self,  out_ch):
        super(outconv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_ch, 1)

    def call(self, x):
        out = self.conv(x)
        logging.debug('outconv: {}->{}'.format(x.shape[1:], out.shape[1:]))
        return out


