__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 03/03/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.5.1"


import tensorflow as tf
from .unet_parts import *


class UNet(tf.keras.Model):
    def __init__(self,n_classes,
                 height, width,
                 known_n_points=None, ngf=64):
        super(UNet, self).__init__()


        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        self.inc = inconv(ngf)
        self.down1 = down(ngf*2)
        self.down2 = down(ngf*2**2)
        self.down3 = down(ngf*2**3)
        self.down4 = down(ngf*2**3)
        self.down5 = down(ngf*2**3)
        self.down6 = down(ngf*2**3)
        self.down7 = down(ngf*2**3)
        self.down8 = down(ngf*2**3, normaliz=False)
        self.up1 = up(ngf*2**3)
        self.up2 = up(ngf*2**3)
        self.up3 = up(ngf*2**3)
        self.up4 = up(ngf*2**3)
        self.up5 = up(ngf*2**2)
        self.up6 = up(ngf*2)
        self.up7 = up(ngf)
        self.up8 = up(ngf, activ=False)
        self.outc = outconv(n_classes)
        self.out_nonlin = tf.keras.layers.Activation('sigmoid')
        

    def call(self, x):
        batch_size = x.shape[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        x = self.outc(x)
        x = self.out_nonlin(x)

        # Reshape Bx1xHxW -> BxHxW
        # because probability map is real-valued by definition
        x = tf.squeeze(x)
        return x


