import numpy as np
import scipy.signal as signal

from scipy import io
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


def load_data_BCI2a(data_path, subject, training):

    n_channels = 22
    n_tests = 6 * 48
    window_length = 7 * 250

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))

    n_valid_trial = 0

    if training:
        a = io.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = io.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a["data"]
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        # a_fs = a_data3[3]
        # a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        # a_gender = a_data3[6]
        # a_age = a_data3[7]

        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                data_return[n_valid_trial, :, :] = np.transpose(
                    a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_length), :n_channels]
                )
                class_return[n_valid_trial] = int(a_y[trial])
                n_valid_trial += 1

    return data_return[0:n_valid_trial, :, :], class_return[0: n_valid_trial]


def load_data_BCI2b(data_path, subject, training):

    n_channels = 3
    n_tests = 120+120+160
    window_length = 8 * 250

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))

    n_valid_trial = 0

    if training:
        a = io.loadmat(data_path + 'B0' + str(subject) + 'T.mat')
    else:
        a = io.loadmat(data_path + 'B0' + str(subject) + 'E.mat')
    a_data = a["data"]
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        # a_fs = a_data3[3]
        # a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        # a_gender = a_data3[6]
        # a_age = a_data3[7]

        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                data_return[n_valid_trial, :, :] = np.transpose(
                    a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_length), :n_channels]
                )
                class_return[n_valid_trial] = int(a_y[trial])
                n_valid_trial += 1

    return data_return[0:n_valid_trial, :, :], class_return[0: n_valid_trial]


def load_data_loso(data_path, subject, DataSet='BCI2a'):

    X_train, y_train = np.empty(0), np.empty(0)
    for sub in range(0, 9):
        if DataSet=='BCI2a':
            X1, y1 = load_data_BCI2a(data_path, sub + 1, True)
            X2, y2 = load_data_BCI2a(data_path, sub + 1, False)
        else:
            X1, y1 = load_data_BCI2b(data_path, sub + 1, True)
            X2, y2 = load_data_BCI2b(data_path, sub + 1, False)
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if sub == subject:
            X_test = X
            y_test = y
        elif not X_train.any():
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


def standardize_data(X_train, X_test, channels):

    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


def bandpass_filter(data, bandFiltCutF, fs, filtOrder=50, axis=1, filtType='filter'):
    a = [1]

    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] is None or bandFiltCutF[1] == fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data
    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF[1], pass_zero="lowpass", fs=fs)
    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF[0], pass_zero="highpass", fs=fs)
    else:
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF, pass_zero="bandpass", fs=fs)

    if filtType == 'filtfilt':
        dataOut = signal.filtfilt(h, a, data, axis=axis)
    else:
        dataOut = signal.lfilter(h, a, data, axis=axis)

    return dataOut


def get_data(data_path, subject, LOSO=False, isStandard=True, freFilter=False, DataSet='BCI2a'):
    if DataSet == 'BCI2a':
        fs = 250  # sampling rate
        t1 = int(1.5 * fs)  # start samples
        t2 = int(6 * fs)  # end samples
        T = t2 - t1  # length of the MI trial

        # Load and split the dataset into training and testing
        if LOSO:
            # Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach.
            X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, DataSet=DataSet)
        else:
            X_train, y_train = load_data_BCI2a(data_path, subject + 1, True)
            X_test, y_test = load_data_BCI2a(data_path, subject + 1, False)
    else:
        fs = 250  # sampling rate
        t1 = int(2.5 * fs)  # start samples
        t2 = int(7 * fs)  # end samples
        T = t2 - t1  # length of the MI trial

        # Load and split the dataset into training and testing
        if LOSO:
            # Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach.
            X_train, y_train, X_test, y_test = load_data_loso(data_path, subject, DataSet=DataSet)
        else:
            X_train, y_train = load_data_BCI2b(data_path, subject + 1, True)
            X_test, y_test = load_data_BCI2b(data_path, subject + 1, False)

    # Prepare training data
    n_tr, n_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2].reshape(n_tr, 1, n_ch, T)
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)

    # Prepare testing data
    n_test, n_ch, _ = X_test.shape
    X_test = X_test[:, :, t1:t2].reshape(n_test, 1, n_ch, T)
    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)

    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, n_ch)

    # Frequency filter
    if freFilter:
        filtBanks = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        X_train_temp = np.zeros(X_train.shape + (len(filtBanks),))
        X_test_temp = np.zeros(X_test.shape + (len(filtBanks),))
        for i in range(len(filtBanks)):
            X_train_temp[:, :, :, :, i] = bandpass_filter(data=X_train, bandFiltCutF=filtBanks[i], fs=125, axis=-1)
            X_test_temp[:, :, :, :, i] = bandpass_filter(data=X_test, bandFiltCutF=filtBanks[i], fs=125, axis=-1)
        X_train = np.transpose(X_train_temp.squeeze(1), (0, 3, 1, 2))
        X_test = np.transpose(X_test_temp.squeeze(1), (0, 3, 1, 2))

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot



