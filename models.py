from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras import backend as K

class Models:
    def __init__(self, config):
        # Input shape
        self.imsize = config["data"]["imsize"]
        self.imchannels = config["data"]["imchannels"]
        self.imshape = (self.imsize, self.imsize, self.imchannels)

    def build_generator(self, gf=64, network="disentangler", stream="C"):

        def conv2d(layer_input, filters, f_size=3, linear=False, skip_input=None):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            if skip_input:
                d = Concatenate()([d, skip_input])
            return d

        def residual_block(layer_input, filters, f_size=3, skip_input=None):
            shortcut = layer_input
            d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=f_size, padding='same')(d)
            d = BatchNormalization()(d)
            d = Add()([d, shortcut])
            if skip_input:
                d = Concatenate()([d, skip_input])
            return d

        def deconv2d(layer_input, filters, f_size=3):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            u = InstanceNormalization()(u)
            u = Activation("relu")(u)
            return u

        g0 = Input(shape=self.imshape)

        if network == "disentangler":
            inputs = g0
        elif network == "entangler":
            s2 = Input(shape=(gf*2, 4, 4))
            s4 = Input(shape=(gf*4, 4, 4))
            s6 = Input(shape=(gf*4, 4, 4))
            inputs = [g0, s2, s4, s6]

        # Downsampling
        g1 = conv2d(g0, gf, f_size=9)
        g2 = conv2d(g1, gf*2, skip_input=s2 if network == "entangler" else None)

        # Redidual-blocks
        g3 = residual_block(g2, gf*4)
        g4 = residual_block(g3, gf*4, skip_input=s4 if network == "entangler" else None)
        g5 = residual_block(g4, gf*4)
        g6 = residual_block(g5, gf*4, skip_input=s6 if network == "entangler" else None)

        if stream == "C":
            # Upsampling
            g7 = deconv2d(g6, gf*2)
            g8 = deconv2d(g7, gf)

            output = Conv2D(self.imchannels, kernel_size=3, activiation="tanh")(g8)
        elif stream == "R":
            output = [g2, g4, g6]

        return Model(inputs, output)

    def build_discriminator(self, df=64):

        def d_layer(layer_input, filters, f_size=3, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if normalization: 
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        image = Input(shape=self.imshape)

        d1 = d_layer(image, df, normalization=False)
        d2 = d_layer(d1, df*2)
        d3 = d_layer(d2, df*4)
        d4 = d_layer(d3, df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(image, validity)