import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, cohen_kappa_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Preprocessing import get_data


# %%
def draw_learning_curves(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path, DataSet='BCI2a'):
    # Generate confusion matrix plot
    if DataSet == 'BCI2a':
        display_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    else:
        display_labels = ['Left hand', 'Right hand']

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 7

    fig, ax = plt.subplots(figsize=(85 / 25.4, 70 / 25.4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=display_labels)
    disp.plot(ax=ax, cmap='inferno')
    if DataSet == 'BCI2a':
        ax.set_xticklabels(display_labels, rotation=12)
    else:
        ax.set_xticklabels(display_labels)

    if DataSet == 'BCI2a':
        plt.title(r'BCI Competition Ⅳ-2a')
    else:
        plt.title(r'BCI Competition Ⅳ-2b')
    plt.tight_layout()

    plt.savefig(results_path + '/subject_' + sub + '.png', dpi=360)
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])


# %% Training
def train(dataset_conf, datasetFuc_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics for all runs (to calculate average
    # accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    # Get dataset parameters
    n_sub = datasetFuc_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    DataSet = dataset_conf.get('DataSet')
    isStandard = datasetFuc_conf.get('isStandard')
    LOSO = datasetFuc_conf.get('LOSO')
    freFilter = datasetFuc_conf.get('freFilter')
    # Get training hyperparameters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub[1], n_train))
    kappa = np.zeros((n_sub[1], n_train))

    # Iteration over subjects
    for sub in range(n_sub[0], n_sub[1]):
        # n_sub: for all subjects, (i-1, i): for the ith subject, get the current 'IN' time to calculate the subject
        # training time.
        in_sub = time.time()
        print("\nTraining on subject", sub + 1)
        log_write.write('\nTraining on subject ' + str(sub + 1) + '\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0
        bestTrainingHistory = []
        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard, freFilter,
                                                                        DataSet)

        # Iteration over multiple runs. How many repetitions of training for a subject.
        for tra in range(n_train):
            # Get the current "IN" time to calculate the "run" training time
            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + "/saved models/run-{}".format(tra + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + "/subject-{}.h5".format(sub + 1)

            # Create the model
            model = getModel(model_name, dataset_conf)
            # Compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True,
                                save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=epochs,
                                batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate the performance of the trained model. Here we load the Trained weights from the file saved in
            # the hard disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, tra] = accuracy_score(labels, y_pred)
            kappa[sub, tra] = cohen_kappa_score(labels, y_pred)

            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, tra + 1,
                                                                           ((out_run - in_run) / 60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, tra], kappa[sub, tra])
            print(info)
            log_write.write(info + '\n')
            # If current training run is better than previous runs, save the history.
            if BestSubjAcc < acc[sub, tra]:
                BestSubjAcc = acc[sub, tra]
                bestTrainingHistory = history

        best_run = np.argmax(acc[sub, :])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run + 1, sub + 1) + '\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub + 1, best_run + 1,
                                                                              ((out_sub - in_sub) / 60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run],
                                                                          np.average(acc[sub, :]),
                                                                          acc[sub, :].std())
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run],
                                                                           np.average(kappa[sub, :]),
                                                                           kappa[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info + '\n')
        # Plot Learning curves
        if LearnCurves:
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory)

    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Store the accuracy and kappa metrics as arrays for all runs into a .npz file format, which is an uncompressed
    # zipped archive, to calculate average accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc=acc, kappa=kappa)

    # Close open files
    best_models.close()
    log_write.close()
    perf_allRuns.close()


# %% Models
def getModel(model_name, dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    if model_name == 'DBNet':
        model = Model.DBNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)

    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model


# %% Testing
def test(model, dataset_conf, datasetFuc_conf, results_path, allRuns=True):
    # Open the "Log" file to write the evaluation results
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")

    # Get dataset parameters
    n_classes = dataset_conf.get('n_classes')
    n_sub = datasetFuc_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    DataSet = dataset_conf.get('DataSet')
    isStandard = datasetFuc_conf.get('isStandard')
    freFilter = datasetFuc_conf.get('freFilter')
    LOSO = datasetFuc_conf.get('LOSO')

    # Initialize variables
    acc_bestRun = np.zeros(n_sub[1])
    kappa_bestRun = np.zeros(n_sub[1])
    cf_matrix = np.zeros([n_sub[1], n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for all runs (experiments) for each subject.
    if allRuns:
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz file format, which is an
        # uncompressed zipped archive, to calculate average accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']

    # Iteration over subjects, n_sub: for all subjects, (i-1,i): for the ith subject.
    for sub in range(n_sub[0], n_sub[1]):
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard, freFilter, DataSet)
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub + 1), results_path, DataSet)

        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub + 1,
                                                       (filepath[filepath.find('run-') + 4:filepath.find('/sub')]))
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub])
        if allRuns:
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub, :].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub, :].std())
        print(info)
        log_write.write('\n' + info)

    # Print & write the average performance measures for all subjects
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub[1]-n_sub[0], np.average(acc_bestRun), np.average(kappa_bestRun))
    if allRuns:
        info = info + ('\nAverage of {} subjects x {} runs (average of {} experiments):'
                       '\nAccuracy = {:.4f}   Kappa = {:.4f}').format(
            n_sub[1]-n_sub[0], acc_allRuns.shape[1], ((n_sub[1]-n_sub[0]) * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns))
    print(info)
    log_write.write(info)

    # Draw a performance bar chart for all subjects
    draw_performance_barChart(n_sub[1], acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub[1], kappa_bestRun, 'K-score')
    # Draw confusion matrix for all subjects (average)
    # np.savez(results_path + "/cf_matrix.npz", cf_matrix=cf_matrix)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path, DataSet)
    # Close open files
    log_write.close()


# %% set GPU
def set_gpu_device(gpu_index):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU,", "Using cuda:", gpu_index)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices available")


# %% Running
def run(DataSet='BCI2a'):
    # Get a dataset path
    set_gpu_device(5)

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # Set dataset parameters
    # If the model is chosen FBCNet, 'freFilter' must be set to "True"

    datasetFuc_conf = {'isStandard': True, 'LOSO': False, 'freFilter': False, 'n_sub': (0, 9)}

    if DataSet == 'BCI2a':
        data_path = "/data4/louxicheng/EEG_data/Motor_Imagery/BCI_Competition_IV_2a/"
        dataset_conf = {'n_classes': 4, 'n_channels': 22, 'in_samples': 1125, 'data_path': data_path, 'DataSet':'BCI2a'}

    else:
        data_path = "/data4/louxicheng/EEG_data/Motor_Imagery/BCI_Competition_IV_2b/"
        dataset_conf = {'n_classes': 2, 'n_channels': 3, 'in_samples': 1125, 'data_path': data_path, 'DataSet':'BCI2b'}

    # Set training hyperparameters
    train_conf = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009,
                  'LearnCurves': True, 'n_train': 10, 'model': 'DBNet'}

    # Train the model
    train(dataset_conf, datasetFuc_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, datasetFuc_conf, results_path)


# %% main
if __name__ == "__main__":
    run(DataSet='BCI2b')
