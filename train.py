import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
import sys

vgg11 = [['c', 64], ['mp'], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['mp'], ]
vgg13 = [['c', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['mp'], ]
vgg16 = [['c', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['mp'], ]
vgg19 = [['c', 64], ['c', 64], ['mp'], ['c', 128], ['c', 128], ['mp'], ['c', 256], ['c', 256], ['c', 256], ['c', 256], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['c', 512], ['mp'], ['c', 512], ['c', 512], ['c', 512], ['c', 512], ['mp'], ]

VGG_DICT = {
        'vgg11': vgg11, 
        'vgg13': vgg13, 
        'vgg16': vgg16, 
        'vgg19': vgg19, 
        }




def print_stdout_and_file(*args):
    print(*args)
    orig_stdout = sys.stdout
    f = open(os.path.join(MODEL_PATH, 'log.txt'), 'a')
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
    cfm_plot.figure.savefig(os.path.join(MODEL_PATH, aux_name + "_confusionmatrix.png"))

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

    epochs_range = list(range(num_epochs))

    plt.figure(figsize=(8, 8))

    add_graph_line('accuracy', 2, 1, 1)
    add_graph_line('loss', 2, 1, 2)

    plt.savefig(os.path.join(MODEL_PATH, aux_name + "_plot.png"))

class CustomVGG():
    def __init__(self, model_structure, num_classes=10):
        self.num_classes = num_classes
        self.model = self.VGG(model_structure)
        self.model.compile(
            optimizer=sgd,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
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

            elif elem[0] == 'mp':
                aux_ll = [
                    tf.keras.layers.MaxPooling2D(strides=(2, 2)),
                    ]

            else:
                raise Exception("Unknown layer type")

            ll += aux_ll

        ll += [
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(SESSION_ARGS['L2_PENALTY'])),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(SESSION_ARGS['DROPOUT']),

            tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(SESSION_ARGS['L2_PENALTY'])),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(SESSION_ARGS['DROPOUT']),

            tf.keras.layers.Dense(1000),
            #tf.keras.layers.LeakyReLU(alpha=0.05),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(self.num_classes),

            tf.keras.layers.Softmax(),
            ]

        return tf.keras.Sequential(ll)

class CustomDataloader():
    def __init__(self, data_dir, batch_size=1, img_size=(224, 224)):
        AUTOTUNE = tf.data.AUTOTUNE
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.class_names = os.listdir(data_dir)
        self.class_names.sort()
        self.list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, "*", "*.*"))
        self.list_ds = self.list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        self.list_ds = self.list_ds.cache()
        self.list_ds = self.list_ds.shuffle(buffer_size=1000)
        self.list_ds = self.list_ds.batch(batch_size)
        self.list_ds = self.list_ds.prefetch(buffer_size=AUTOTUNE)

    def get_dataset(self):
        return self.list_ds

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        result = tf.argmax(one_hot)
        return result


    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])


    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

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
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history_synth = model.fit(
        train_ds.get_dataset(),
        validation_data=val_ds.get_dataset(),
        epochs=num_epochs,
        workers=SESSION_ARGS['NUM_WORKERS'],
        use_multiprocessing=True,
        callbacks=[model_checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)

    model.save_weights(os.path.join(MODEL_PATH, "model_" + aux_write + ".h5"))

    generate_graph(history_synth, num_epochs, aux_write)
    generate_confusion_matrix(model, test_ds.get_dataset(), aux_write, train_ds.class_names)

    return model

def main():
    model = CustomVGG(model_structure=VGG_DICT[SESSION_ARGS['VGG_MODEL']], num_classes=SESSION_ARGS['NUM_CLASSES']).model

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
            optimizer=sgd,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
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
    SESSION_ARGS = {
        'NUM_CLASSES': 10,
        'MODEL_PATH': None,
        'BATCH_SIZE': 2**7,
        'NUM_WORKERS': 10,
        'EPOCHS_REAL': 40,
        'EPOCHS_SYNTH': 15,
        'VGG_MODEL': None,
        'DROPOUT': 0.5,
        'IMG_SIZE': (224, 224),
        'TRAIN_TYPE': 'hybrid', # hybrid or REAL or SYNTH
        'TRAIN_DIR_REAL' : './images/real/train/',
        'VAL_DIR_REAL' : './images/real/val/',
        'TEST_DIR_REAL' : './images/test/',
        'TRAIN_DIR_SYNTH' : './images/synth/train/',
        'VAL_DIR_SYNTH' : './images/synth/val/',
        'TEST_DIR_SYNTH' : './images/test/',
        'LEARNING_RATE' : 10**-2,
        'MOMENTUM': 0.9,
        'L2_PENALTY': 5*10**-4,
    }

    MODEL_PATH = 'MODEL_' + f"{datetime.now()}"
    MODEL_PATH = MODEL_PATH.replace(' ', '_')
    MODEL_PATH = MODEL_PATH.replace(':', '_')
    MODEL_PATH = MODEL_PATH.replace('.', '_')
    MODEL_PATH = MODEL_PATH.replace('-', '_')
    SESSION_ARGS['MODEL_PATH'] = MODEL_PATH

    nargs = len(sys.argv) - 1

    for arg_idx in range(1, nargs+1):
        if arg_idx % 2 == 1:
            arg_type = sys.argv[arg_idx]
            arg_value = sys.argv[arg_idx + 1]

            if arg_type in [
                    'EPOCHS_REAL', 
                    'EPOCHS_SYNTH', 
                    'BATCH_SIZE', 
                    'NUM_WORKERS',
                    ]:
                SESSION_ARGS[arg_type] = int(arg_value)

            elif arg_type in [
                    'DROPOUT',
                    'LEARNING_RATE',
                    'MOMENTUM',
                    'L2_PENALTY',
                    ]:
                SESSION_ARGS[arg_type] = float(arg_value)

            elif arg_type == 'TRAIN_TYPE':
                
                if arg_value in ['hybrid', 'real', 'synth']:
                    SESSION_ARGS[arg_type] = arg_value

                else:
                    raise Exception("Unknown TRAIN_TYPE " + arg_value + " only types allowed are hybrid or REAL or SYNTH")

            else:
                SESSION_ARGS[arg_type] = arg_value

    if SESSION_ARGS['VGG_MODEL'] is None:
        raise Exception("The VGG model has not been specified. The available types are " + str(list(VGG_DICT.keys())))

    os.mkdir(MODEL_PATH)

    for key, value in SESSION_ARGS.items():
        print_stdout_and_file(key, ' : ', value)

    sgd = tf.keras.optimizers.SGD(learning_rate=SESSION_ARGS['LEARNING_RATE'], momentum=SESSION_ARGS['MOMENTUM'], nesterov=False, name='SGD')

    main()
