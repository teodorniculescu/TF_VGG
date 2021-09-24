import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys
from helper import *

def print_stdout_and_file(*args):
    print(*args)
    orig_stdout = sys.stdout
    f = open(os.path.join(SESSION_ARGS['MODEL_PATH'], 'log.txt'), 'a')
    sys.stdout = f

    print(*args)

    sys.stdout = orig_stdout
    f.close()

def generate_confusion_matrix(model, ds, aux_name, classes):
    num_classes = len(classes)
    y_true = []
    y_pred = []
    print_stdout_and_file("generating confusion matrix")
    X_train = list(map(lambda x: x[0], ds))
    y_train = list(map(lambda x: x[1], ds))
    for idx in range(len(X_train)):
        X = X_train[idx]
        y = y_train[idx].numpy()[0]
        p = np.argmax(np.array(model.predict(X)[0]))
        y_true.append(y)
        y_pred.append(p)
    cfm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.tight_layout()
    cfm_plot.figure.savefig(os.path.join(SESSION_ARGS['MODEL_PATH'], aux_name + "_confusionmatrix.png"))

def generate_graph(history, num_epochs, aux_name):
    print_stdout_and_file("generating graph")
    def add_graph_line(label, nrows, ncols, nidx):
        label_data = history.history[label]
        val_label_data = history.history['val_' + label]
        plt.subplot(nrows, ncols, nidx)
        plt.plot(epochs_range, label_data, label='Training ' + label)
        plt.plot(epochs_range, val_label_data, label='Validation ' + label)
        #plt.legend(loc='upper right')
        plt.legend(loc='best')
        plt.title('Training and Validation ' + label)

    print(history.history)

    epochs_range = list(range(num_epochs))

    plt.figure(figsize=(8, 8))

    add_graph_line('accuracy', 2, 1, 1)
    add_graph_line('loss', 2, 1, 2)

    plt.savefig(os.path.join(SESSION_ARGS['MODEL_PATH'], aux_name + "_plot.png"))

def run(model, train_dir, val_dir, test_dir, aux_write, num_epochs):
    train_dir = './images/real/train/'
    val_dir = './images/real/val/'
    test_dir = './images/test/'

    train_ds =  CustomDataloader(train_dir, batch_size=SESSION_ARGS['BATCH_SIZE'])
    val_ds = CustomDataloader(val_dir, batch_size=SESSION_ARGS['BATCH_SIZE'])
    test_ds = CustomDataloader(test_dir)

    if not (train_ds.class_names == 
            val_ds.class_names == 
            test_ds.class_names):
        raise Exception("The test validation and train sets don't have the same classes or the classes are not in the same order!" +
        str(test_ds.class_names) + 
        str(val_ds.class_names) + 
        str(train_ds.class_names))

    checkpoint_filepath = '/tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history_synth = model.fit(
        train_ds.get_dataset(),
        validation_data=val_ds.get_dataset(),
        epochs=num_epochs,
        #workers=SESSION_ARGS['NUM_WORKERS'],
        use_multiprocessing=True,
        callbacks=[model_checkpoint_callback],
    )

    model = tf.keras.models.load_model(checkpoint_filepath)

    model.save(os.path.join(SESSION_ARGS['MODEL_PATH'], "model_" + aux_write + ".h5"))

    generate_graph(history_synth, num_epochs, aux_write)

    generate_confusion_matrix(model, test_ds.get_dataset(), aux_write + "_test", train_ds.class_names)
    trds = CustomDataloader(SESSION_ARGS['TRAIN_DIR_REAL']).get_dataset()
    generate_confusion_matrix(model, trds, aux_write + "_train", train_ds.class_names)
    vds = CustomDataloader(SESSION_ARGS['VAL_DIR_REAL']).get_dataset()
    generate_confusion_matrix(model, vds, aux_write + "_val", train_ds.class_names)

    return model

def main():
    if not os.path.exists(SESSION_ARGS['MODEL_PATH']):
        os.mkdir(SESSION_ARGS['MODEL_PATH'])

    print_stdout_and_file('The executed command is: ' + " ".join(sys.argv))

    for key, value in SESSION_ARGS.items():
        print_stdout_and_file(key, ' : ', value)

    model = CustomVGG(model_structure=VGG_DICT[SESSION_ARGS['VGG_MODEL']], num_classes=SESSION_ARGS['NUM_CLASSES'], optimizer=SESSION_ARGS['OPTIMIZER'], loss=SESSION_ARGS['LOSS'], metrics=SESSION_ARGS['METRICS'], l2_penalty=SESSION_ARGS['L2_PENALTY'], dropout=SESSION_ARGS['DROPOUT'], input_shape=SESSION_ARGS['IMG_SIZE']).model

    if SESSION_ARGS['TRAIN_TYPE'] in ['hybrid', 'synth']:
        run(
            model,        
            SESSION_ARGS['TRAIN_DIR_SYNTH'],
            SESSION_ARGS['VAL_DIR_SYNTH'],
            SESSION_ARGS['TEST_DIR_SYNTH'],
            'synth',
            SESSION_ARGS['EPOCHS_SYNTH']
        )

    if SESSION_ARGS['TRAIN_TYPE'] in ['hybrid', 'real']:
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.4, fill_mode='reflect'),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
            model,
            ])

        model.compile(
            optimizer=SESSION_ARGS['OPTIMIZER'],
            loss=SESSION_ARGS['LOSS'],
            metrics=SESSION_ARGS['METRICS'],
            )

        run(
            model,        
            SESSION_ARGS['TRAIN_DIR_REAL'],
            SESSION_ARGS['VAL_DIR_REAL'],
            SESSION_ARGS['TEST_DIR_REAL'],
            'real',
            SESSION_ARGS['EPOCHS_REAL']
        )


if __name__ == "__main__":
    SESSION_ARGS = get_session_args()

    # Start main execution

    main()
