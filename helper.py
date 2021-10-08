import tensorflow as tf
import cv2
import argparse
from datetime import datetime
import numpy as np
import os

vgg11 = [['cis', 64], ['mp'], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['mp'], ]
vgg13 = [['cis', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['mp'], ]
vgg16 = [['cis', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['mp'], ]
vgg19 = [['cis', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['c', 512], ['mp'], ]

VGG_DICT = {
        'vgg11': vgg11, 
        'vgg13': vgg13, 
        'vgg16': vgg16, 
        'vgg19': vgg19, 
        }

def get_session_args(args=None):
    SESSION_ARGS = {
        'NUM_CLASSES': 10,
        'MODEL_PATH': None,
        'BATCH_SIZE': 2**7,
        #'NUM_WORKERS': 10,
        'EPOCHS_REAL': 40,
        'EPOCHS_SYNTH': 15,
        'VGG_MODEL': None,
        'DROPOUT': 0.5,
        'IMG_SIZE': (224, 224, 3),
        'TRAIN_TYPE': 'hybrid', # hybrid or real or synth 
        'TRAIN_DIR_REAL' : './images/real/train/',
        'VAL_DIR_REAL' : './images/real/val/',
        'TEST_DIR_REAL' : './images/test/',
        'TRAIN_DIR_SYNTH' : './images/synth/train/',
        'VAL_DIR_SYNTH' : './images/synth/val/',
        'TEST_DIR_SYNTH' : './images/test/',
        'LEARNING_RATE' : 10**-2,
        'MOMENTUM': 0.9,
        'L2_PENALTY': 5*10**-4,
        'METRICS': [
            'accuracy',
            #tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            #tf.keras.metrics.Precision(name='precision'),
            #tf.keras.metrics.Recall(name='recall'),
            ],
        'LOSS': 'sparse_categorical_crossentropy',
        'OPTIMIZER': None,
    }

    # Initialize argparse

    parser = argparse.ArgumentParser(description='Train VGG models.')
    parser.add_argument('--MODEL_PATH', type=str, help='The path where the model results will be saved')
    parser.add_argument('--BATCH_SIZE', type=int, help='The batch size')
    parser.add_argument('--EPOCHS_REAL', type=int, help='The number of epochs executed for training the model on real data.')
    parser.add_argument('--EPOCHS_SYNTH', type=int, help='The number of epochs executed for training the model on synthetic data.')
    parser.add_argument('--VGG_MODEL', type=str, help='The type of the model vgg model, currently available models are ' + str(VGG_DICT.keys()) + '.')
    parser.add_argument('--DROPOUT', type=float, help='The rate of the dropout layer.')
    parser.add_argument('--TRAIN_TYPE', type=str, help='The type of training data used for training the VGG model. Available values are synth, real and hybrid. If you use real, the model is only trained on real data. If you use synth, the model is only trained on synth data. If you use hybrid, the model is trained on both real and synthetic data.')
    parser.add_argument('--TRAIN_DIR_REAL', type=str, help='The directory path of the real images used for training.')
    parser.add_argument('--TRAIN_DIR_SYNTH', type=str, help='The directory path of the synthetic images used for training.')
    parser.add_argument('--VAL_DIR_REAL', type=str, help='The directory path of the real images used for validation.')
    parser.add_argument('--VAL_DIR_SYNTH', type=str, help='The directory path of the synthetic images used for validation.')
    parser.add_argument('--TEST_DIR_REAL', type=str, help='The directory path of the real images used for testing.')
    parser.add_argument('--TEST_DIR_SYNTH', type=str, help='The directory path of the synthetic images used for testing.')
    parser.add_argument('--LEARNING_RATE', type=float, help='The learning rate of the model.')
    parser.add_argument('--MOMENTUM', type=float, help='The momentum of the model.')
    parser.add_argument('--L2_PENALTY', type=float, help='The l2 penalty of the model.')

    # Get values from argparse and put them in the SESSION ARGUMENTS
    if args is None:
        opt = vars(parser.parse_args())
    else:
        opt = vars(parser.parse_args(args))
    for key in opt.keys():
        if key in SESSION_ARGS:
            if opt[key] is not None:
                SESSION_ARGS[key] = opt[key]
        else:
            raise Exception("The key " + key + " does not exists in the SESSION_ARGS!")

    # Initialize anything that was left as 'None'

    if SESSION_ARGS['MODEL_PATH'] is None:
        MODEL_PATH = 'MODEL_' + f"{datetime.now()}"
        MODEL_PATH = MODEL_PATH.replace(' ', '_')
        MODEL_PATH = MODEL_PATH.replace(':', '_')
        MODEL_PATH = MODEL_PATH.replace('.', '_')
        MODEL_PATH = MODEL_PATH.replace('-', '_')
        SESSION_ARGS['MODEL_PATH'] = MODEL_PATH


    # SESSION ARGUMENTS that require other SESSION ARGUMENTS in order to be initialized.

    SESSION_ARGS['OPTIMIZER'] = tf.keras.optimizers.SGD(learning_rate=SESSION_ARGS['LEARNING_RATE'], momentum=SESSION_ARGS['MOMENTUM'], nesterov=False, name='SGD')

    return SESSION_ARGS

class CustomVGG():
    def __init__(self, model_structure, num_classes=10, optimizer=None, loss=None, metrics=None, l2_penalty=None, dropout=None, input_shape=None):
        self.input_shape=input_shape
        self.num_classes = num_classes
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.model = self.VGG(model_structure)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            )

    def VGG(self, t):
        ll = []

        for elem in t:

            if elem[0] == 'c':
                aux_ll = [
                    tf.keras.layers.Conv2D(elem[1], 3, padding='same'),
                    #tf.keras.layers.LeakyReLU(alpha=0.05),
                    tf.keras.layers.ReLU(),
                    ]

            elif elem[0] == 'cis':
                aux_ll = [
                    tf.keras.layers.Conv2D(elem[1], 3, padding='same', input_shape=self.input_shape),
                    #tf.keras.layers.LeakyReLU(alpha=0.05),
                    tf.keras.layers.ReLU(),
                    ]

            elif elem[0] == 'mp':
                aux_ll = [
                    tf.keras.layers.MaxPooling2D(strides=(2, 2)),
                    ]

            else:
                raise Exception("Unknown layer type")

            ll += aux_ll

        ll += [
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(self.l2_penalty)),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(self.dropout),

            tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(self.l2_penalty)),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(self.dropout),

            tf.keras.layers.Dense(1000),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(self.num_classes),

            tf.keras.layers.Softmax(),
            ]

        return tf.keras.Sequential(ll)

class CustomDataloader():
    def __init__(self, data_dir, img_size=(224, 224)):
        AUTOTUNE = tf.data.AUTOTUNE
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.class_names = os.listdir(data_dir)
        self.class_names.sort()
        images_list = []
        labels_list = []
        
        for idx, cls in enumerate(self.class_names):
            for image_name in os.listdir(os.path.join(data_dir, cls)):
                image_path = os.path.join(data_dir, cls, image_name)
                images_list.append(self.process_path(image_path))
                labels_list.append(idx)

        self.list_ds = (np.array(images_list), np.array(labels_list))

    def get_dataset(self):
        return self.list_ds

    def process_path(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.resize(img, (self.img_width, self.img_height))
        return img
