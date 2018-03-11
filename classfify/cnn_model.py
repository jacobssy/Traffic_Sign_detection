from cnn_util import conv,pool,dropout
from keras.layers import Dense,Input, Flatten,concatenate
from keras.models import Model
IMG_SIZE =48
NUM_CLASSES =43
def inception_model(x):
    layer1 = conv(x, 16, (3, 3), padding='same', strides=(2, 2), name='layer1')
    layer2 = conv(layer1, 48, (5, 5), padding='same', strides=(1, 1), name='layer2')
    layer3 = conv(layer2, 48, (3, 3), padding='same', strides=(2, 2), name='layer3')
    layer3_pool = pool(layer3,pool_size=(3, 3), strides=(1, 1), padding='same',pool_type='max')
    layer4 = dropout(layer3_pool,rate = 0.2)

    branch1_1 = conv(layer4, 64, (3, 3), padding='same', strides=(2, 2), name='branch1_1')
    branch1_2 = conv(branch1_1, 64, (5, 5), padding='same', strides=(1, 1), name='branch1_2')

    branch2_1 = conv(layer4, 64, (5, 5), padding='same', strides=(1, 1), name='branch2_1')
    branch2_2 = conv(branch2_1, 64, (7, 7), padding='same', strides=(2, 2), name='branch2_2')

    branch3_1 = pool(layer4,pool_size=(5, 5), strides=(1, 1), padding='same', name='branch3_1',pool_type='max')
    branch3_2 = conv(branch3_1, 64, (5, 5), padding='same', strides=(2, 2), name='branch3_2')

    x = concatenate([branch1_2, branch2_2, branch3_2], axis=3)
    return x

def cnn_model():
    data_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inception_model(data_input)
    x = conv(x, 128, (3, 3), padding='same', strides=(1, 1), name='layer5')
    x = conv(x, 256, (3, 3), padding='same', strides=(2, 2), name='layer6')
    x = pool(x,pool_size=(3, 3), strides=(1, 1), padding='same', name='layer7',pool_type='max')
    x = dropout(x,rate=0.5)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    x = Model(data_input, x, name='inception')
    return x