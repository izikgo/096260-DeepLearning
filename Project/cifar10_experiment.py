"""
CIFAR10 experiment for Doubly Convolutional Layer
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.backend import image_data_format
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.layers import ZeroPadding2D, Dense
from keras.backend import cast_to_floatx
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from doubly_convolutional import DoubleConv2D

batch_size = 200
optimizer = 'adadelta'
validation_size = 10000

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = cast_to_floatx(X_train)
X_test = cast_to_floatx(X_test)

# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, random_state=0)

# preprocess
row_axis, col_axis, channel_axis = (1, 2, 3) if image_data_format() == 'channels_last' else (2, 3, 1)
train_mean = np.mean(X_train, axis=(0, row_axis, col_axis))
broadcast_shape = [1, 1, 1]
broadcast_shape[channel_axis - 1] = X_train.shape[channel_axis]
train_mean = np.reshape(train_mean, broadcast_shape)
X_train -= train_mean
# X_val -= train_mean
X_test -= train_mean


def create_model(use_doubly=False):
    if use_doubly:
        conv_layer_args = {'filters': 128, 'kernel_size': (4, 4), 'effective_kernel_size': (3, 3), 'pool_size': (2, 2),
                           'activation': 'relu'}
        ConvLayer = DoubleConv2D
    else:
        conv_layer_args = {'filters': 128, 'kernel_size': 3, 'activation': 'relu'}
        ConvLayer = Conv2D

    model = Sequential()

    for i in range(4):
        if i == 0:
            model.add(ZeroPadding2D(padding=(1, 1), input_shape=X_train.shape[1:]))
        else:
            model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(ConvLayer(**conv_layer_args))
        model.add(BatchNormalization(axis=channel_axis))

        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(ConvLayer(**conv_layer_args))
        model.add(BatchNormalization(axis=channel_axis))

        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(Y_train.shape[-1], activation='softmax'))

    return model

conv_model = create_model(use_doubly=False)
conv_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10)
checkpoint_cb = ModelCheckpoint(filepath="regular_conv_model.{epoch:02d}-{val_acc:.4f}.hdf5", verbose=1,
                                monitor='val_acc', save_best_only=True)
conv_model.fit(X_train, Y_train, batch_size=batch_size, epochs=150, validation_data=(X_test, Y_test),
               callbacks=[reduce_lr_cb, checkpoint_cb])

# print(conv_model.evaluate(X_test, Y_test, batch_size=batch_size))
