from __future__ import print_function

import numpy as np
#from keras.models import Model
#from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, LeakyReLU

from keras_layer_AnchorBoxes_tf import AnchorBoxes


import numpy as np
import warnings

#import sys
#sys.path.insert(0, '/home/hicham/keras_1_2_0/')

from keras import backend as K
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.advanced_activations import *
from keras.layers.pooling import *
from keras.activations import *
from keras.layers.convolutional import *
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.constraints import *
from keras.layers.noise import *

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

from keras_layer_L2Normalization import L2Normalization
from keras_layer_PostProcessing import PostProcessing

tf = K.tf
dropout_rate = 0.55
W_regularizer = None
init_ = 'glorot_uniform'
conv_has_bias = True #False for BN
fc_has_bias = True

OMP_NUM_THREADS= 4

expand_bias = False

#######################################################################

#Depthwise convolution 

def relu6(x):
    return K.relu(x, max_value=6)

class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3 #tf
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


def _depthwise_conv_block_detection(input, layer_name, strides = (1,1), 
                          kernel_size = 3,  
                          pointwise_conv_filters=32, alpha=1.0, depth_multiplier=1, 
                          padding = 'valid', 
                          data_format = None, 
                          activation = None, use_bias = True, 
                          depthwise_initializer='glorot_uniform', 
                          pointwise_initializer='glorot_uniform', bias_initializer = "zeros",  
                          bias_regularizer= None, activity_regularizer = None, 
                          depthwise_constraint = None, pointwise_constraint = None,  
                          bias_constraint= None, batch_size = None, 
                          block_id=1,trainable = None, weights = None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((kernel_size, kernel_size),
                        padding=padding,
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=expand_bias,
                        name=layer_name + '_conv_dw_%d' % block_id)(input)
    x = BatchNormalization(axis=channel_axis, name=layer_name + '_conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name=layer_name+'_conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               #padding='same',
               padding=padding, 
               use_bias=expand_bias,
               strides=(1, 1),
               name=layer_name + '_conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,  name=layer_name+'_conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6,  name=layer_name+ '_conv_pw_%d_relu' % block_id)(x)

def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
 
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=expand_bias,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=expand_bias,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
  
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=expand_bias,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

########################################################################
class Scaling(Layer):
    def __init__(self, init_weights=1.0, bias=True, trainable=True, **kwargs):
        self.supports_masking = True
        self.initial_weights = init_weights
        self.has_bias = bias
        self.trainable = trainable
        super(Scaling, self).__init__(**kwargs)

    def build(self, input_shape):
        size = input_shape[-1]  # Tensorflow
        self.scaling_factor = self.add_weight(shape=(1, 1, size), initializer='one', trainable=self.trainable, name=None)

        if (self.has_bias):
            self.bias = self.add_weight(shape=(1, 1, size), initializer='zero', trainable=self.trainable, name= None )
            self.trainable_weights = [self.scaling_factor, self.bias]
        else:
            self.trainable_weights = [self.scaling_factor]

    def call(self, x, mask=None):
        out = self.scaling_factor * x
        if (self.has_bias):
            out = out + self.bias
        return out

    def get_config(self):
        config = {}
        config['scaling_factor'] = self.scaling_factor
        if (self.has_bias):
            config['bias'] = self.bias
        base_config = super(Scaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def bn_conv(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample =(1,1), border_mode ='same', bias=conv_has_bias):
    tmp_layer = input_layer
    #tmp_layer = Conv2D(nb_filter, kernel_size = (nb_row, nb_col), strides = subsample, padding = border_mode, name = layer_name, bias = bias, kernel_initializer=init_, kernel_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, activation=None, border_mode=border_mode, name = layer_name, bias=bias, init=init_, W_regularizer= W_regularizer)(tmp_layer)
    tmp_layer = BatchNormalization(name = layer_name + '_bn')(tmp_layer)
    tmp_layer = Lambda(lambda x:tf.nn.relu(x), name = layer_name + '_nonlin')(tmp_layer)
    return tmp_layer


def bn_conv_layer(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample=(1,1), border_mode = 'same',bias=conv_has_bias):
    tmp_layer = input_layer
    #tmp_layer = Conv2D(nb_filter, kernel_size = (nb_row, nb_col), strides = subsample, padding = border_mode, name = layer_name, bias = bias, kernel_initializer=init_, kernel_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample, activation=None, border_mode=border_mode, name=layer_name, bias=bias, init=init_, W_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = Scaling(name=layer_name + '_scale')(tmp_layer)
    tmp_layer = Lambda(lambda x: tf.nn.elu(x), name=layer_name + '_nonlin')(tmp_layer)
    return tmp_layer

def add_inception(input_layer, list_nb_filter, base_name):
    tower_1_1 = bn_conv_layer(input_layer=input_layer, layer_name=base_name + '_1x1', nb_filter=list_nb_filter[0], nb_row=1, nb_col=1)

    tower_2_1 = bn_conv_layer(input_layer=input_layer, layer_name=base_name + '_3x3_reduce', nb_filter=list_nb_filter[1], nb_row=1, nb_col=1)
    tower_2_2 = bn_conv_layer(input_layer=tower_2_1, layer_name=base_name + '_3x3', nb_filter=list_nb_filter[2], nb_row=3, nb_col=3)

    tower_3_1 = bn_conv_layer(input_layer=input_layer, layer_name=base_name + '_5x5_reduce',nb_filter=list_nb_filter[3],nb_row=1, nb_col=1)
    tower_3_2 = bn_conv_layer(input_layer=tower_3_1, layer_name=base_name + '_5x5', nb_filter=list_nb_filter[4], nb_row=5, nb_col=5)

    tower_4_1 = MaxPooling2D(name=base_name + '_pool',pool_size=(3, 3), strides=(1, 1), border_mode='same')(input_layer)
    tower_4_2 = bn_conv_layer(input_layer=tower_4_1, layer_name=base_name + '_pool_proj', nb_filter=list_nb_filter[5],nb_row=1, nb_col=1)

    return merge(inputs=[tower_1_1,tower_2_2,tower_3_2,tower_4_2], mode='concat',name=base_name + '_output')

def bn_conv_layer_detection(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample=(1,1), border_mode = 'same',bias=conv_has_bias):
    tmp_layer = input_layer
    #tmp_layer = Conv2D(nb_filter, kernel_size = (nb_row, nb_col), strides = subsample, padding = border_mode, name = layer_name, bias = bias, kernel_initializer=init_, kernel_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample, activation=None, border_mode=border_mode, name=layer_name, bias=bias, init=init_, W_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = BatchNormalization(name=layer_name + '_bn')(tmp_layer)
    tmp_layer = Lambda(lambda x: tf.nn.relu(x), name=layer_name + '_nonlin')(tmp_layer)
    return tmp_layer


def preprocess_input(x):
    x -= 128.0
    return x

def preprocess_input_inv(x):
    x += 128.0
    return x


def DetectionModel():

    img_height = 512
    img_width = 512
    img_channels = 3
    min_scale = 0.1
    max_scale = 0.9 
    n_classes = 2
    #scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # anchorboxes for coco dataset

    aspect_ratios_per_layer = [[0.5, 1.0, 2.0],
                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                     [1.0/3.0, 0.5, 1.0, 2.0, 3.0], 
                     [0.5, 1.0, 2.0],
                     [0.5, 1.0, 2.0]] # The anchor box aspect ratios used in the original SSD300
    two_boxes_for_ar1 = True
    limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
    coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
    normalize_coords = False
    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300

# Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    aspect_ratios_conv4_3 = [0.5, 1.0, 2.0]
    aspect_ratios_fc7     = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]
    aspect_ratios_conv6_2 = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]
    aspect_ratios_conv7_2 = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]
    aspect_ratios_conv8_2 = [0.5, 1.0, 2.0]
    aspect_ratios_conv9_2 = [0.5, 1.0, 2.0]

 # Compute the number of boxes to be predicted per cell for each predictor layer.
# We need this so that we know how many channels the predictor layers need to have.

    n_boxes_conv4_3 = 4 # 4 boxes per cell for the original implementation
    n_boxes_fc7     = 6 # 6 boxes per cell for the original implementation
    n_boxes_conv6_2 = 6 # 6 boxes per cell for the original implementation
    n_boxes_conv7_2 = 6 # 6 boxes per cell for the original implementation
    n_boxes_conv8_2 = 4 # 4 boxes per cell for the original implementation
    n_boxes_conv9_2 = 4 # 4 boxes per cell for the original implementation

    # Input image format
 
    input_shape = (img_height, img_width, img_channels) 

    img_input = Input(shape=input_shape)

    alpha = 1.0
    depth_multiplier = 1


    x = Lambda(lambda z: z/255., # Convert input feature range to [-1,1]
              output_shape=(img_height, img_width, img_channels),
               name='lambda1')(img_input)
    x = Lambda(lambda z: z - 0.5, # Convert input feature range to [-1,1]
                  output_shape=(img_height, img_width, img_channels),
                   name='lambda2')(x)
    x = Lambda(lambda z: z *2., # Convert input feature range to [-1,1]
                  output_shape=(img_height, img_width, img_channels),
                   name='lambda3')(x)

    x = _conv_block(x, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block_classification(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=10)
    conv4_3 = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=11) #11 conv4_3 (300x300)-> 19x19 

    x = _depthwise_conv_block_classification(conv4_3, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)   # (300x300) -> 10x10 
    fc7 = _depthwise_conv_block_classification(x, 1024, alpha, depth_multiplier, block_id=13) # 13 fc7 (300x300) -> 10x10


    conv6_1 = bn_conv(fc7, 'detection_conv6_1', 256, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    conv6_2 = _depthwise_conv_block_detection(input = conv6_1, layer_name='detection_conv6_2', strides=(2,2), 
                                        pointwise_conv_filters=512, alpha=alpha, depth_multiplier=depth_multiplier, 
                                        padding = 'same', use_bias = True, block_id=1)
 
    conv7_1 = bn_conv(conv6_2, 'detection_conv7_1', 128, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    conv7_2 = _depthwise_conv_block_detection(input = conv7_1, layer_name='detection_conv7_2', strides=(2,2), 
                                        pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier, 
                                        padding = 'same', use_bias = True, block_id=2)
    #conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='detection_conv7_1')(conv6_2)
    #conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='detection_conv7_2')(conv7_1)

    conv8_1 = bn_conv(conv7_2, 'detection_conv8_1', 128, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    
    conv8_2 = _depthwise_conv_block_detection(input = conv8_1, layer_name='detection_conv8_2', strides=(2,2), 
                            pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier, 
                            padding = 'same', use_bias = True, block_id=3)

    # # conv8_2 = bn_conv(conv8_1, 'detection_conv8_2', 256, 2, 2, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
        
    conv9_1 = bn_conv(conv8_2, 'detection_conv9_1', 64, 1, 1,  subsample =(1,1), border_mode ='same', bias=conv_has_bias)    
    # conv9_2 = bn_conv(conv9_1, 'detection_conv9_2', 128, 3, 3, subsample =(2,2), border_mode ='same', bias=conv_has_bias)

    conv9_2 = _depthwise_conv_block_detection(input = conv9_1, layer_name='detection_conv9_2', strides=(2,2), 
                                    pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=4)


    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='detection_conv4_3_norm')(conv4_3)

    
    conv4_3_norm_mbox_conf = _depthwise_conv_block_detection(input = conv4_3_norm, layer_name='detection_conv4_3_norm_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv4_3 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=1)
        

    fc7_mbox_conf = _depthwise_conv_block_detection(input = fc7, layer_name='detection_fc7_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_fc7 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=2)
    conv6_2_mbox_conf = _depthwise_conv_block_detection(input = conv6_2, layer_name='detection_conv6_2_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv6_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=3)

    conv7_2_mbox_conf = _depthwise_conv_block_detection(input = conv7_2, layer_name='detection_conv7_2_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv7_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=4)
    
    conv8_2_mbox_conf = _depthwise_conv_block_detection(input = conv8_2, layer_name='detection_conv8_2_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv8_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=5)
    conv9_2_mbox_conf = _depthwise_conv_block_detection(input = conv9_2, layer_name='detection_conv9_2_mbox_conf', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv9_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=6)
    
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`

    conv4_3_norm_mbox_loc = _depthwise_conv_block_detection(input = conv4_3_norm, layer_name='detection_conv4_3_norm_mbox_loc', strides=(1,1), 
                                    pointwise_conv_filters=n_boxes_conv4_3 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                    padding = 'same', use_bias = True, block_id=1) 
    
    fc7_mbox_loc = _depthwise_conv_block_detection(input = fc7, layer_name='detection_fc7_mbox_loc', strides=(1,1), 
                                pointwise_conv_filters=n_boxes_fc7 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                padding = 'same', use_bias = True, block_id=2)
        

    conv6_2_mbox_loc = _depthwise_conv_block_detection(input = conv6_2, layer_name='detection_conv6_2_mbox_loc', strides=(1,1), 
                                pointwise_conv_filters=n_boxes_conv6_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                padding = 'same', use_bias = True, block_id=3)

    conv7_2_mbox_loc = _depthwise_conv_block_detection(input = conv7_2, layer_name='detection_conv7_2_mbox_loc', strides=(1,1), 
                                pointwise_conv_filters=n_boxes_conv7_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                padding = 'same', use_bias = True, block_id=4)

    conv8_2_mbox_loc = _depthwise_conv_block_detection(input = conv8_2, layer_name='detection_conv8_2_mbox_loc', strides=(1,1), 
                                pointwise_conv_filters=n_boxes_conv8_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                padding = 'same', use_bias = True, block_id=5)
    
    conv9_2_mbox_loc = _depthwise_conv_block_detection(input = conv9_2, layer_name='detection_conv9_2_mbox_loc', strides=(1,1), 
                                pointwise_conv_filters=n_boxes_conv9_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier, 
                                padding = 'same', use_bias = True, block_id=5)
    ### Generate the anchor boxes 

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`

  

  # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =0,this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios_conv4_3,
                                             two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =1,this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios_fc7,
                                    two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_fc7_mbox_priorbox')(fc7)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =2,this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios_conv6_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv6_2_mbox_priorbox')(conv6_2)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =3,this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios_conv7_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv7_2_mbox_priorbox')(conv7_2)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =4, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios_conv8_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv8_2_mbox_priorbox')(conv8_2)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, layer_cnt =5, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios_conv9_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv9_2_mbox_priorbox')(conv9_2)
   
    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='detection_fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='detection_mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='detection_mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='detection_mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='detection_mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='detection_predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    confidence_thresh = 0.3
    iou_threshold = 0.20
    
    predictions_decoded = PostProcessing(img_height, img_width, confidence_thresh=confidence_thresh, 
      iou_threshold=iou_threshold, top_k=25, input_coords='centroids', normalize_coords= False)(predictions)


    model = Model(inputs=img_input, outputs=predictions_decoded)
    #model = Model(inputs=img_input, outputs=predictions)
    # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
    # be able to generate the default boxes for the matching process outside of the model during training.
    # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
    # Instead, we'll do it in the batch generator function.
    # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.
    predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                 fc7_mbox_conf._keras_shape[1:3],
                                 conv6_2_mbox_conf._keras_shape[1:3],
                                 conv7_2_mbox_conf._keras_shape[1:3],
                                 conv8_2_mbox_conf._keras_shape[1:3],
                                 conv9_2_mbox_conf._keras_shape[1:3]])

    
    return model

