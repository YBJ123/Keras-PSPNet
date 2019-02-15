#coding=utf-8
from __future__ import print_function
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
import keras.backend as K
import tensorflow as tf
from config import num_classes,img_rows,img_cols

def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # resized = tf.image.resize_nearest_neighbor(inputs, self.new_size)
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
         return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config

def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)
    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad, name=names[2], use_bias=False)(prev)
    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)

    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4], use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj", "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev

def empty_branch(prev):
    return prev

# 对捷径(输入)进行下采样且升维后，然后再残差连接的模块
def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    # 卷积
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)
    # 捷径
    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

# 直接与捷径(输入)进行残差连接的模块
def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)
    # 卷积
    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    # 捷径
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added

# ResNet
# 以inp的尺寸为(473,473,3)为例
def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)
    # "conv1_1_3x3_s2"
    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)   # 输出尺寸：237x237x64（下采样1次）
    bn1 = BN(name=names[1])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "conv1_2_3x3"
    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1) # 输出尺寸：237x237x64
    bn1 = BN(name=names[3])(cnv1)        
    relu1 = Activation('relu')(bn1)       
    # "conv1_3_3x3"
    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],use_bias=False)(relu1) # 输出尺寸：237x237x128
    bn1 = BN(name=names[5])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "pool1_3x3_s2"
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)                     # 输出尺寸：119x119x128(下采样2次）

    # Residual layers(body of network)
    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """
    # conv2_x,ResNet50/101的结构相同，即conv2_1-conv2_3
    # 保持原ResNet不改动
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)                                           # 输出尺寸：119x119x256
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)                                   # 输出尺寸：119x119x256
    
    # conv3_x,ResNet50/101的结构相同，即conv3_1-conv3_3
    # 保持原ResNet不改动
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)                       # 输出尺寸：60x60x512(下采样3次）
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)                                   # 输出尺寸：60x60x512
    
    # conv4_x,ResNet50/101的结构不同
    # 若ResNet50，conv4_1 - conv4_6
    # 改动之处：conv4_1中第一个1x1卷积的步长由2改为1，且conv4_x中的3x3卷积采用膨胀速率为2的膨胀卷积
    if layers is 50:
        # 改动之处1:conv4_1的步长由2改为1，且其中的3x3卷积采用膨胀速率为2的膨胀卷积                                                              
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                       # 输出尺寸：60x60x1024(不下采样)            
        for i in range(5):
            # 改动之处2：conv4_2-conv4_6中的3x3卷积采用膨胀速率为2的膨胀卷积 
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)                               # 输出尺寸：60x60x1024       
    # 若ResNet101，conv4_1 - conv4_23
    # 改动之处：conv4_1的步长由2改为1，且conv4_x中的3x3卷积采用膨胀速率为2的膨胀卷积
    elif layers is 101:
        # 改动之处1：conv4_1的步长由2改为1，且其中的3x3卷积采用膨胀速率为2的膨胀卷积       
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)    
        # 改动之处2：conv4_2-conv4_23中的3x3卷积采用膨胀速率为2的膨胀卷积                                  
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")
    
    # conv5_x,ResNet50/101的结构相同，即conv5_1 - conv5_3
    # 改动之处：conv5_1中第一个1x1卷积的步长由2改为1，且conv5_x中的3x3卷积采用膨胀速率为4的膨胀卷积
    # 改动之处1：conv5_1中第一个1x1卷积的步长由2改为1，且其中的3x3卷积采用膨胀速率为4的膨胀卷积 
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)                                           # 输出尺寸：60x60x2048(不下采样)
    for i in range(2):
        # 改动之处2：conv5_2-conv5_3中的3x3卷积采用膨胀速率为4的膨胀卷积  
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)                                   # 输出尺寸：60x60x2048

    res = Activation('relu')(res)
    return res

# 金字塔池化模块中的层级操作
# 若input_shape为(473,473)，则feature_map_shape为(60,60)
def interp_block(prev_layer, level, feature_map_shape, input_shape):
    ### 依据不同的输入尺寸，定义各层级的平均池化操作的核大小及步长
    if input_shape == (473, 473):
        kernel_strides_map = {1: 60, 2: 30, 3: 20, 6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90, 2: 45, 3: 30, 6: 15}
    # 针对(320,320)，自定义金字塔池化模块
    elif input_shape == (320,320):                                  
        kernel_strides_map = {1: 40, 2: 20, 4: 10, 8: 5}        
    # 针对(512, 512)，自定义金字塔池化模块
    elif input_shape == (512, 512):
        kernel_strides_map = {1: 64, 2: 32, 4: 16, 8: 8}
    else:
        print("Pooling parameters for input shape ", input_shape, " are not defined.")
        exit(1)

    names = ["conv5_3_pool" + str(level) + "_conv", 
             "conv5_3_pool" + str(level) + "_conv_bn"]

    # 获取指定金字塔层级的平均池化的核大小
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    # 获取指定金字塔层级的平均池化的步长大小
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    # 进行平均池化
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    
    # 通过1x1卷积，降通道数，至512
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    
    # 将特征图的空间尺寸还原至金字塔池化模块的输入的空间尺寸，即(60,60,512)
    prev_layer = Interp(feature_map_shape)(prev_layer)

    return prev_layer

# 金字塔池化模块
# 若input_shape=(473,473)，则res的维度(60,60,2048)
def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    
    # 通过input_shape，计算出金字塔尺寸模块的输入特征的空间空间尺寸，即input_shape的1/8
    # feature_map_size = (60,60)
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    
    # 层级1：对res进行平均池化到1x1x2048，降通道数至512，缩放回60x60x512
    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)                  # 输出尺寸为(60,60，512)
    # 层级2：对res进行平均池化到2x2x2048，降通道数至512，缩放回60x60x512
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)                  # 输出尺寸为(60,60，512)
    # 层级3：对res进行平均池化到3x3x2048，降通道数至512，缩放回60x60x512
    interp_block3 = interp_block(res, 4, feature_map_size, input_shape)                  # 输出尺寸为(60,60，512)
    # 层级4：对res进行平均池化到6x6x2048，降通道数至512，缩放回60x60x512
    interp_block6 = interp_block(res, 8, feature_map_size, input_shape)                  # 输出尺寸为(60,60，512)

    # 将四个层级的特征图以及金字塔池化模块的输入相串联，输出尺寸为(60,60,(512x4+2048))=(60,60,4096)
    res = Concatenate()([res,interp_block6,interp_block3,interp_block2,interp_block1])   # 输出尺寸为(60,60,4096)

    return res

def build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols)):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s and predicting %i classes" % (
        resnet_layers, input_shape, num_classes))

    # 定义输入尺寸
    inp = Input((input_shape[0], input_shape[1], 3))                          # 输入尺寸：(473,473,3)

    # 调用施加了膨胀卷积且经预训练的ResNet，输出特征图的空间尺寸为输入的1/8
    res = ResNet(inp, layers=resnet_layers)                                   # 输出尺寸：(60,60,2048)
    
    # 金字塔池化模块
    psp = build_pyramid_pooling_module(res, input_shape)                      # 输出尺寸：(60,60,4096)

    # 通过1x1卷积，降通道数至512
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",   # 输出尺寸：(60,60,512)
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # 通过1x1卷积，调整维度等于目标分类数num_class
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), name="conv6")(x)           # 输出尺寸：(60,60,num_class)

    # 缩放特征图的空间尺寸至输入的空间尺寸
    x = Interp([input_shape[0], input_shape[1]])(x)                           # 输出尺寸：(473,473,num_class)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model

if __name__ == '__main__':
    with tf.device("/cpu:0"):
        pspnet_model = build_pspnet(3, resnet_layers=50, input_shape=(512,512))
        pspnet_model.summary()
    K.clear_session()
