import h5py
import glob
import numpy as np
from skimage import io
from img_util import preprocess_img,get_class
import pandas as pd
import os
NUM_CLASSES = 43
def get_train_data():
    try:
        with h5py.File('train_images.h5') as hf:
            X,Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from train_images.h5")

    except (IOError, OSError, KeyError):
        print("Error in reading train_images.h5. Processing all images...")
        root_dir = '/data/sunsiyuan/2018learning/Traffic_Sign_data/GTSRB/Final_Training/Images'
        imgs = []
        labels = []
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))

                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)
                if len(imgs) % 1000 == 0:
                    print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('the data dir has missed', img_path)
                pass
        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('train_images.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)
    return X,Y
def get_test_data():
    try:
        with  h5py.File('test_images.h5') as hf:
            X_test, y_test = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from test_images.h5")
    except (IOError, OSError, KeyError):
        print("Error in reading test_images.h5. Processing all images...")
        test = pd.read_csv('/data/sunsiyuan/2018learning/Traffic_Sign_data/GTSRB/Final_Test/Images/GT-final_test.csv',sep=';')
        X_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('/data/sunsiyuan/2018learning/Traffic_Sign_data/GTSRB/Final_Test/Images/',file_name)
            X_test.append(preprocess_img(io.imread(img_path)))
            y_test.append(class_id)

        X_test = np.array(X_test, dtype='float32')
        y_test = np.array(y_test, dtype='uint8')

        with h5py.File('test_images.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X_test)
            hf.create_dataset('labels', data=y_test)
    return X_test,y_test

def prepare_data(is_normalize=False):
    index = np.zeros(1307, dtype='int')
    for i in range(1307):
        index[i] = i * 30 + np.random.randint(0, 30)
    X ,Y = get_train_data()
    X_test ,y_test =get_test_data()

    if is_normalize:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    X_val = X[index]
    y_val = Y[index]
    # creat the training index1

    index1 = np.setdiff1d(np.array(range(39209)), index, assume_unique=True)
    X_train = X[index1]
    y_train = Y[index1]

    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape
    return X_train, y_train, X_val, y_val, X_test, y_test