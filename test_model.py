import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Input, Permute
from tensorflow.keras.models import Model

import models
import numpy as np
import attention_models

bands = 1
input_layer = Input((bands, 22, 1125))
# input_layer = Input((16, 1, 70))
# X_test = np.random.rand(1, bands, 22, 1125)

# These models can be chosen: ATCNet, TCNet_Fusion, EEGTCNet, EEGNet_classifier, EEGNeX, DeepConvNet, ShallowConvNet,
# MBEEG_SENet, MI_EEGNet, EEG_Inception, EEG_ITNet, FBCNet, LMDA_Net, MANet

output = models.MANet()(input_layer)
# output = models.MHA_block(input_layer)
model = Model(inputs=input_layer, outputs=output)
# output = model.predict(X_test)
model.summary()
# print(output)
