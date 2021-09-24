import argparse
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
import sys
from helper import *

def evaluate(ds, model, aux):
    X_train = np.array(list(map(lambda x: x[0][0], ds)))
    y_train = np.array(list(map(lambda x: x[1], ds)))
    results = model.evaluate(X_train, y_train, verbose=1)
    print(aux, 'loss', results[0], 'acc', results[1])


if __name__ == "__main__":
    args = '--VGG_MODEL vgg11 --TRAIN_TYPE real --EPOCHS_REAL 2 --LEARNING_RATE 0.001 --L2_PENALTY 0.00005'.split()
    SESSION_ARGS = get_session_args(args)

    #model = CustomVGG(model_structure=VGG_DICT[SESSION_ARGS['VGG_MODEL']], num_classes=SESSION_ARGS['NUM_CLASSES'], optimizer=SESSION_ARGS['OPTIMIZER'], loss=SESSION_ARGS['LOSS'], metrics=SESSION_ARGS['METRICS'], l2_penalty=SESSION_ARGS['L2_PENALTY'], dropout=SESSION_ARGS['DROPOUT'], input_shape=SESSION_ARGS['IMG_SIZE']).model

    #model.built = True
    checkpoint_filepath = "MODEL_2021_09_24_20_41_11_419015/model_real.h5"
    model = tf.keras.models.load_model(checkpoint_filepath)

    evaluate( CustomDataloader(SESSION_ARGS['TRAIN_DIR_REAL']).get_dataset() , model, 'train')
    evaluate( CustomDataloader(SESSION_ARGS['VAL_DIR_REAL']).get_dataset() , model, 'val')
    evaluate( CustomDataloader(SESSION_ARGS['TEST_DIR_REAL']).get_dataset() , model, 'test')
