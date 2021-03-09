from keras.datasets import cifar10
from keras.layers import Dense, Flatten, Conv2D, add
from keras.layers import Activation, BatchNormalization, Input, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import optimizers, regularizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train_mean = np.mean(x_train, axis=(0, 1, 2))
x_train_std = np.std(x_train, axis=(0, 1, 2))
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std

# Convert class vectors to binary class matrices.
y_train_onehot = np_utils.to_categorical(y_train, 10)
y_test_onehot = np_utils.to_categorical(y_test, 10)


def scheduler(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 51, 101, 201 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        learning rate(float32)
    """
    if epoch < 51:
        return 0.1
    if epoch < 101:
        return 0.01
    if epoch < 201:
        return 0.001
    return 0.0001


# Parameter of Regularizers.
weight_decay = 1e-4

input = Input(shape=(32, 32, 3))

# Build model.
# Regularizers allow to apply penalties on layer parameters or layer activity during optimization. 
# These penalties are incorporated in the loss function that the network optimizes.
# To avoid overfitting.
x = Conv2D(16, (1, 1), padding='same', input_shape=(32, 32, 3), strides=(1, 1),
             kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(input)
for _ in range(5):
    res1 = BatchNormalization()(x)
    res1 = Activation('elu')(res1)
    res1 = Conv2D(16, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res1)
    res2 = BatchNormalization()(res1)
    res2 = Activation('elu')(res2)
    res2 = Conv2D(16, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res2)
    x = add([res2, x])
layer1 = BatchNormalization()(x)
layer1 = Activation('elu')(layer1)
# layer1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer1)


x = Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(layer1)


layer2 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(x)
res1 = BatchNormalization()(layer2)
res1 = Activation('elu')(res1)
res1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res1)
res2 = BatchNormalization()(res1)
res2 = Activation('elu')(res2)
res2 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res2)
x = add([res2, x])


for _ in range(1, 5):
    res1 = BatchNormalization()(x)
    res1 = Activation('elu')(res1)
    res1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res1)
    res2 = BatchNormalization()(res1)
    res2 = Activation('elu')(res2)
    res2 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res2)
    x = add([res2, x])
layer3 = BatchNormalization()(x)
layer3 = Activation('elu')(layer3)
# layer3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer3)


x = Conv2D(64, (3, 3), padding='same', strides=(2, 2),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(layer3)


layer4 = Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(x)
res1 = BatchNormalization()(layer4)
res1 = Activation('elu')(res1)
res1 = Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res1)
res2 = BatchNormalization()(res1)
res2 = Activation('elu')(res2)
res2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res2)
x = add([res2, x])


for _ in range(1, 5):
    res1 = BatchNormalization()(x)
    res1 = Activation('elu')(res1)
    res1 = Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res1)
    res2 = BatchNormalization()(res1)
    res2 = Activation('elu')(res2)
    res2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                  kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='he_normal')(res2)
    x = add([res2, x])
layer5 = BatchNormalization()(x)
layer5 = Activation('relu')(layer5)
layer5 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer5)


classifier = Flatten()(layer5)
classifier = Dense(10)(classifier)
classifier = Activation('softmax')(classifier)

# Instantiate model.
model = Model(input, classifier)
try:
    # Load best model weights.
    model.load_weights("ResNet_best-model.hdf5")
    print("Load model Successfully!")
except:
    print("Creating a new model.")
model.summary()
opt = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy',
                   metrics=['accuracy'])

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # randomly rotate images in the range
    rotation_range=10,
    # randomly shift images horizontally
    width_shift_range=0.2,
    # randomly shift images vertically
    height_shift_range=0.2,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False)
# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint("ResNet_best-model.hdf5", monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max', period=1)
change_lr = LearningRateScheduler(scheduler)

# Run training, with data augmentation.
# train_history = model.fit_generator(datagen.flow(x_train, y_train_onehot, batch_size=64),
#                                     epochs=500, shuffle=True,
#                                     validation_data=(x_test, y_test_onehot),
#                                     steps_per_epoch=x_train.shape[0]//32,
#                                     verbose=2, callbacks=[checkpoint, change_lr])


# Plot the result of training.
def show_train_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title("Train History")
  plt.ylabel(train)
  plt.xlabel("Epoch")
  plt.legend(['train','validation'], loc='upper left')
  plt.show()


show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')


# Testing.
model.load_weights("ResNet_best-model.hdf5")
score = model.evaluate(x_test, y_test_onehot)
print("Testing accuracy=", score[1]*100, "%")

prediction = model.predict(x_test)
prediction = prediction.argmax(axis=-1)
y_test = y_test.reshape(10000)
print("Predict :", prediction[0:10])
print("Real :", y_test[0:10])

import pandas as pd
confuse = pd.crosstab(y_test.reshape(-1), prediction, rownames=['label'],
                      colnames=['predict'])
print(confuse)
