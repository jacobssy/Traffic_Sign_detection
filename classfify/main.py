from dataloader import prepare_data
from cnn_model import cnn_model
from keras import backend as K
import numpy as np
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from time import  time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
# def my_init(shape, name=None):
#     value = np.random.random(shape)
#     return K.variable(value, name=name)


def train():
    lr = 0.002
    batch_size = 16
    epoch = 80
    start = time()
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model = cnn_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    datagen = ImageDataGenerator(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 width_shift_range=0.15,
                                 height_shift_range=0.15,
                                 zoom_range=0.2,
                                 shear_range=0.15,
                                 rotation_range=10., )
    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=2000,
                            epochs=epoch,
                            validation_data=(X_val, y_val),
                            callbacks=[ReduceLROnPlateau('val_loss', factor=0.2, patience=20, verbose=1, mode='auto'),
                                       ModelCheckpoint('model.h5',save_best_only=True)]
                           )
    end = time()
    print (end - start)

def test():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    start = time()
    model = load_model('model.h5')
    pred = model.predict(X_test,batch_size=1000)
    end=time()
    y_pred = np.empty(12630)
    for i in range(0,12630):
        y_pred[i] = (np.argmax(pred[i][:]))
    acc = np.mean(y_pred==y_test)
    print("Test accuracy = {}".format(acc))
    print  (end-start)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] =='test':
        test()
