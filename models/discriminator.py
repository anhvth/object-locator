import tensorflow as tf 

class DiscrimConv(tf.keras.Model):
    def __init__(self, out_channels, stride):
        super(DiscrimConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_channels, 4, strides=(stride, stride), padding='valid')

    def call(self, x):
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        x = self.conv(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        n_layers = 3
        with tf.variable_scope("layer_1"):
            layers = [DiscrimConv(ndf, stride=2)]
            layers += [tf.keras.layers.LeakyReLU(.2)]

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                layers += [DiscrimConv(out_channels, stride=stride), 
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(.2)]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            layers += [DiscrimConv(out_channels=1, stride=1), 
                    tf.keras.layers.Activation('sigmoid')]

            self.conv = tf.keras.models.Sequential(layers)

        
    def call(self, inputs, targets):
        x = tf.concat([inputs, targets], axis=-1)
        x = self.conv(x)
        return x
        


if __name__ == '__main__':
    dis_input = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    dis_target = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    dis_model = Discriminator(ndf=32)
    dis_out = dis_model(dis_input, dis_target)
    print('created discriminator:', dis_out.shape)
    dis_model.conv.summary()
