from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda,add,UpSampling2D,concatenate
from keras.layers import  Add, Concatenate, GlobalAveragePooling2D,GlobalMaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import numpy as np
from keras.regularizers import l2
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs

################################################################################


class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class FullYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        return image / 255.

class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        return image / 255.

class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))
        mobilenet = MobileNet(input_shape=(480,480,3), include_top=False,weights=None)
        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image		

class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image    

class Inception3Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))
        inception = InceptionV3(input_shape=(input_size,input_size,3), include_top=False,weights=None)
        x = inception(input_image)
        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image

class VGG16Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))
        vgg16 = VGG16(input_shape=(input_size, input_size, 3), include_top=False)
        self.feature_extractor = Model(input_image,x)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image 

class ResNet50Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        resnet50 = ResNet50(input_shape=(input_size, input_size, 3), include_top=False,weights=None)
        resnet50.layers.pop() # remove the average pooling layer
        self.feature_extractor = Model(resnet50.layers[0].input, resnet50.layers[-1].output)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image

class YoloV3Feature(BaseFeatureExtractor):
    def __init__(self,input_size):
        input_image = Input(shape=(input_size, input_size, 3))
        #layer1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=True)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer2
        x = Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', use_bias=True)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer3
        shortcut = Activation('linear')(x)
        x = Conv2D(32, (1,1), strides=(1,1), padding='same', name='conv_3', use_bias=True)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer4
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=True)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer5
        x = Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_5', use_bias=True)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer6
        shortcut = Activation('linear')(x)
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_6', use_bias=True)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer7
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=True)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer8
        shortcut = Activation('linear')(x)
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_8', use_bias=True)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer9
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=True)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = add([x,shortcut])
        #layer10
        x = Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_10', use_bias=True)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer11 - layer26
        for i in range(0,8):
            shortcut = Activation('linear')(x)
            x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+11), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+11))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+12), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+12))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])
        skip1 = x
        #layer27
        x = Conv2D(512, (3,3), strides=(2,2), padding='same', name='conv_27', use_bias=True)(x)
        x = BatchNormalization(name='norm_27')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer28-layer44
        for i in range(0,8):
            shortcut = Activation('linear')(x)
            x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+28), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+28))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+29), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+29))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])
        #layer45
        skip2 = x
        x = Conv2D(1024, (3,3), strides=(2,2), padding='same', name='conv_45', use_bias=True)(x)
        x = BatchNormalization(name='norm_45')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer46 - layer53
        for i in range(0,4):
            shortcut = Activation('linear')(x)
            x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_'+str(2*i+46), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+46))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_'+str(2*i+47), use_bias=True)(x)
            x = BatchNormalization(name='norm_'+str(2*i+47))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = add([x,shortcut])

        # layer 54 ,55 , output: 75 = 3*(4+1+20),it should be adapted by dataset
        yolov3_1 = x
        yolov3_1 = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_54', use_bias=True)(yolov3_1)
        yolov3_1 = BatchNormalization(name='norm_54')(yolov3_1)
        yolov3_1 = LeakyReLU(alpha=0.1)(yolov3_1)
        yolov3_1 = Conv2D(75, (1,1), strides=(1,1), padding='same', name='conv_55', use_bias=True)(yolov3_1)
        yolov3_1 = Activation('linear')(yolov3_1)
        # layer 56
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_56', use_bias=True)(x)
        x = BatchNormalization(name='norm_56')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip2])
        #layer 57
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_57', use_bias=True)(x)
        x = BatchNormalization(name='norm_57')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer 58
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_58', use_bias=True)(x)
        x = BatchNormalization(name='norm_58')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer 59
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_59', use_bias=True)(x)
        x = BatchNormalization(name='norm_59')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer 60
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_60', use_bias=True)(x)
        x = BatchNormalization(name='norm_60')(x)
        x = LeakyReLU(alpha=0.1)(x)
        # layer 61
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_61', use_bias=True)(x)
        x = BatchNormalization(name='norm_61')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #layer 62
        yolov3_2 = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_62', use_bias=True)(x)
        yolov3_2 = BatchNormalization(name='norm_62')(yolov3_2)
        yolov3_2 = LeakyReLU(alpha=0.1)(yolov3_2)
        #layer 63
        yolov3_2 = Conv2D(75, (1,1), strides=(1,1), padding='same', name='conv_63', use_bias=True)(yolov3_2)
        #layer 64
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_64', use_bias=True)(x)
        x = BatchNormalization(name='norm_64')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D(2)(x)
        x = concatenate([x,skip1])
        #layer 65
        yolov3_3 = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_65', use_bias=True)(x)
        yolov3_3 = BatchNormalization(name='norm_65')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 66
        yolov3_3 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_66', use_bias=True)(yolov3_3)
        yolov3_3 = BatchNormalization(name='norm_66')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 67
        yolov3_3 = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_67', use_bias=True)(yolov3_3)
        yolov3_3 = BatchNormalization(name='norm_67')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 68
        yolov3_3 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_68', use_bias=True)(yolov3_3)
        yolov3_3 = BatchNormalization(name='norm_68')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 69
        yolov3_3 = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_69', use_bias=True)(yolov3_3)
        yolov3_3 = BatchNormalization(name='norm_69')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 70
        yolov3_3 = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_70', use_bias=True)(yolov3_3)
        yolov3_3 = BatchNormalization(name='norm_70')(yolov3_3)
        yolov3_3 = LeakyReLU(alpha=0.1)(yolov3_3)
        # layer 71
        yolov3_3 = Conv2D(75, (1,1), strides=(1,1), padding='same', name='conv_71', use_bias=True)(yolov3_3)

        # feature_extractor
        self.feature_extractor = Model(input_image,[yolov3_1,yolov3_2,yolov3_3], name="yolov3")

    def normalize(self, image):
        return image / 255.

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[0][1:3]

class S_MobileNet(BaseFeatureExtractor):
    def __init__(self,input_size):
        input_image = Input(shape=(input_size,input_size,3))
        # x = conv_block(input_image, 32, weight_decay=0.01, name='conv1', strides=(2, 2))
        x = InvertedResidualBlock(input_image, expand=4, out_channels=32, repeats=2, stride=2, weight_decay=0.01, block_id=1)
        x = InvertedResidualBlock(x, expand=4, out_channels=48, repeats=2, stride=2, weight_decay=0.01, block_id=2)
        x = shuffle_unit(x, 48, 96, 4, 0.25, strides=2, stage=1, block=3)
        x = InvertedResidualBlock(x, expand=2, out_channels=128, repeats=3, stride=1, weight_decay=0.01, block_id=4)
        x = InvertedResidualBlock(x, expand=2, out_channels=192, repeats=3, stride=1, weight_decay=0.01, block_id=5)
        x = shuffle_unit(x,192,256,4,0.25, strides=2, stage=1, block=6)
        x = fire_module(x,7, squeeze=64, expand=256)
        x = fire_module(x,8, squeeze=48, expand=196)
        x = fire_module(x, 9, squeeze=32, expand=128)
        x = fire_module(x, 10, squeeze=16, expand=64)
        x = fire_module(x, 11, squeeze=4, expand=16,strides=2)
        self.feature_extractor = Model(input_image,x,name="S_MobileNet")

    def normalize(self, image):
        return image / 255.


def fire_module(x, fire_id, squeeze=16, expand=64,strides =1):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    s_id = 'fire' + str(fire_id) + '/'

    x = Conv2D(squeeze, (1, 1), strides = (strides,strides),padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')
    return x

def shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    """
    creates a shuffleunit
    Parameters
    ----------
    inputs:
        Input tensor of with `channels_last` data format
    in_channels:
        number of input channels
    out_channels:
        number of output channels
    strides:
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    block: int(1)
        block number
    Returns
    -------
    """
    prefix = 'stage%d/block%d' % (stage, block)

    #if strides >= 2:
        #out_channels -= in_channels

    # default: 1/4 of the output channel of a ShuffleNet Unit
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    x = group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                    groups=(1 if stage == 2 and block == 1 else groups),
                    name='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(axis=-1, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    x = Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
                        strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=-1, name='%s/bn_dwconv_1' % prefix)(x)

    x = group_conv(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels,
                    groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(axis=-1, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        ret = Concatenate(-1, name='%s/concat' % prefix)([x, avg])

    ret = Activation('relu', name='%s/relu_out' % prefix)(ret)

    return ret

def group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """
    grouped convolution
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    in_channels:
        number of input channels
    out_channels:
        number of output channels
    groups:
        number of groups per channel
    kernel: int(1)
        An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    stride: int(1)
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for all spatial dimensions.
    name: str
        A string to specifies the layer name
    Returns
    -------
    """
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride, name=name)(x)

    # number of intput channels per group
    ig = in_channels // groups
    group_list = []

    assert out_channels % groups == 0

    for i in range(groups):
        offset = i * ig
        group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                 use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
    return Concatenate(name='%s/concat' % name)(group_list)

def channel_shuffle(x, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    Examples
    --------
    Example for a 1D Array with 3 groups
    >>> d = np.array([0,1,2,3,4,5,6,7,8])
    >>> x = np.reshape(d, (3,3))
    >>> x = np.transpose(x, [1,0])
    >>> x = np.reshape(x, (9,))
    '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])

    return x

def Relu6(x, **kwargs):
    return Activation(relu6, **kwargs)(x)

def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):
    '''
    This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.
    :param x: Input Keras tensor in (B, H, W, C_in)
    :param expand: expansion factor in bottlenect residual block
    :param out_channels: number of channels in the output tensor
    :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.
    :param stride: stride for the 1x1 convolution
    :param weight_decay: hyperparameter for the l2 penalty
    :param block_id: as its name tells
    :return: Output tensor (B, H_new, W_new, out_channels)
    '''
    channel_axis = -1
    in_channels = K.int_shape(x)[channel_axis]
    x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,
                kernel_regularizer=l2(weight_decay), name='conv_%d_0' % block_id)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_1' % block_id)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=1,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='conv_dw_%d_0' % block_id )(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_dw_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_2' % block_id)
    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
               kernel_regularizer=l2(weight_decay), name='conv_bottleneck_%d_0' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_0_bn' % block_id)(x)

    for i in xrange(1, repeats):
        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='conv_%d_%d' % (block_id, i))(x)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name='conv_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1,name='conv_%d_%d_act_1' % (block_id, i))
        x1 = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=1,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name='conv_dw_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9, name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1, name='conv_dw_%d_%d_act_2' % (block_id, i))
        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay),name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
        x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
    return x

def conv_block(inputs, filters, weight_decay, name, kernel=(3, 3), strides=(1, 1)):
    '''
    Normal convolution block performs conv+bn+relu6 operations.
    :param inputs: Input Keras tensor in (B, H, W, C_in)
    :param filters: number of filters in the convolution layer
    :param name: name for the convolutional layer
    :param kernel: kernel size
    :param strides: strides for convolution
    :return: Output tensor in (B, H_new, W_new, filters)
    '''
    channel_axis = -1
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               strides=strides,
               name=name)(inputs)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name=name+'_bn')(x)
