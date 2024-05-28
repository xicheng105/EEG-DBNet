from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, Conv1D,
                                     Conv2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, Reshape,
                                     Flatten, Add, Concatenate, Input, Permute, multiply, GlobalAveragePooling1D)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


def LC_Block(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25, activation='elu', AveragePooling=True):
    conv_block1 = Conv2D(F1, kernel_size=(1, kernLength), padding='same', data_format='channels_first',
                         use_bias=False)(input_layer)
    conv_block1 = BatchNormalization(axis=1)(conv_block1)

    conv_block2 = DepthwiseConv2D(kernel_size=(Chans, 1), depth_multiplier=D, data_format='channels_first',
                                  use_bias=False, depthwise_constraint=max_norm(1.))(conv_block1)
    conv_block2 = BatchNormalization(axis=1)(conv_block2)
    conv_block2 = Activation(activation)(conv_block2)
    if AveragePooling:
        conv_block2 = AveragePooling2D(pool_size=(1, kernLength / 8), data_format='channels_first')(conv_block2)
    else:
        conv_block2 = MaxPooling2D(pool_size=(1, kernLength / 8), data_format='channels_first')(conv_block2)
    conv_block2 = Dropout(dropout)(conv_block2)

    conv_block3 = SeparableConv2D(F1*D, kernel_size=(1, kernLength // 4), padding='same', data_format='channels_first',
                                  use_bias=False)(conv_block2)
    conv_block3 = BatchNormalization(axis=1)(conv_block3)
    conv_block3 = Activation(activation)(conv_block3)
    if AveragePooling:
        conv_block3 = AveragePooling2D(pool_size=(1, kernLength / 8), data_format='channels_first')(conv_block3)
    else:
        conv_block3 = MaxPooling2D(pool_size=(1, kernLength / 8), data_format='channels_first')(conv_block3)

    conv_block3 = Dropout(dropout)(conv_block3)
    conv_block3 = K.squeeze(conv_block3, axis=-2)

    return conv_block3

def SE_Block(input_layer, Seize=2, activation1='relu', activation2='sigmoid', BandSE=True):
    if BandSE: # (32, 17)
        aver_block = GlobalAveragePooling1D(data_format='channels_first')(input_layer) # (32)
        bands = input_layer.shape[1]
        aver_block = Reshape((1, bands))(aver_block) # (1, 32)
        se_block = Dense(Seize, activation=activation1, kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')(aver_block) # (1, 2)
        se_block = Dense(bands, activation=activation2, kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')(se_block) # (1, 32)
        se_block = Permute((2, 1))(se_block) # (32, 1)
    else: # (16, 31)
        aver_block = GlobalAveragePooling1D(data_format='channels_last')(input_layer) # (31)
        timePoints = input_layer.shape[2]
        aver_block = Reshape((1, timePoints))(aver_block) # (1, 31)
        se_block = Dense(Seize, activation=activation1, kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')(aver_block) # (1, 2)
        se_block = Dense(timePoints, activation=activation2, kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')(se_block) # (1, 31)

    se_block = multiply([input_layer, se_block])

    return se_block

def GC_Block(input_layer, dropout=0.3, depth=2, activation='elu', kernel_size=4, TimeConv=True, n_windows=5, step=4,
             seize=2):
    F1 = input_layer.shape[1]
    F2 = input_layer.shape[2]
    sw_concat = []

    if TimeConv:  # (16, 31)
        for j in range(n_windows):
            st = j * step
            end = F2 - (n_windows - j - 1) * step
            sw = input_layer[:, :, st:end]
            se_block = SE_Block(input_layer=sw, BandSE=False, Seize=seize, activation1='relu')
            last_block = se_block
            for i in range(depth):
                block_1 = Conv1D(F1, kernel_size=kernel_size, dilation_rate=i+1, padding='causal',
                                 kernel_initializer='he_uniform', data_format='channels_first')(last_block)
                block_1 = BatchNormalization(axis=1)(block_1)
                block_1 = Activation(activation)(block_1)
                block_1 = Dropout(dropout)(block_1)
                add_block = Add()([block_1, se_block])
                last_block = Activation(activation)(add_block)
            fl_block = Flatten()(last_block)
            sw_concat.append(fl_block)
        ca_block = Concatenate()(sw_concat)

    else: # (32, 17)
        for j in range(n_windows):
            st = j * step
            end = F1 - (n_windows - j - 1) * step
            sw = input_layer[:, st:end, :]
            se_block = SE_Block(input_layer=sw, BandSE=True, Seize=seize, activation1='relu')
            last_block = se_block
            for i in range(depth):
                block_1 = Conv1D(F2, kernel_size=kernel_size, dilation_rate=i+1, padding='causal',
                                 kernel_initializer='he_uniform', data_format='channels_last')(last_block)
                block_1 = BatchNormalization(axis=-1)(block_1)
                block_1 = Activation(activation)(block_1)
                block_1 = Dropout(dropout)(block_1)
                add_block = Add()([block_1, se_block])
                last_block = Activation(activation)(add_block)
            fl_block = Flatten()(last_block)
            sw_concat.append(fl_block)
        ca_block = Concatenate()(sw_concat)

    return ca_block

def EEG_DBNet(nb_classes=4, Chans=22, Samples=1125, regRate=0.25, d=4, k=4, n=6, s=1, se=2):
    inputs = Input(shape=(1, Chans, Samples))

    LC_Block1 = LC_Block(input_layer=inputs, F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu',
                         AveragePooling=True) # (16, 31)
    LC_Block2 = LC_Block(input_layer=inputs, F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu',
                         AveragePooling=False) # (32, 17)

    GC_Block1 = GC_Block(input_layer=LC_Block1, TimeConv=True, depth=d, kernel_size=k, n_windows=n, step=s,
                         seize=se, activation='elu')
    GC_Block2 = GC_Block(input_layer=LC_Block2, TimeConv=False, depth=d, kernel_size=k, n_windows=n, step=s,
                         seize=se, activation='elu')

    conc_block = Concatenate()([GC_Block1, GC_Block2]) # 512

    dense_block = Dense(nb_classes, kernel_constraint=max_norm(regRate))(conc_block)
    softmax = Activation('softmax')(dense_block)

    return Model(inputs=inputs, outputs=softmax)

