from dataloader import prepare_data
from cnn_model import cnn_model
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from time import  time
from keras.preprocessing.image import ImageDataGenerator
# def my_init(shape, name=None):
#     value = np.random.random(shape)
#     return K.variable(value, name=name)


def train():
    lr = 0.002
    batch_size = 16
    epoch = 60
    start = time()
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model = cnn_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10., )
    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=1000,
                            epochs=epoch,
                            validation_data=(X_val, y_val),
                            callbacks=[ReduceLROnPlateau('val_loss', factor=0.2, patience=20, verbose=1, mode='auto'),
                                       ModelCheckpoint('model.h5',save_best_only=True)]
                           )
    end = time()
    print (end - start)

def test():
    pass


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
train()
