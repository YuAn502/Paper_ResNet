import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, add, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

InitSeed = 767
tf.random.set_seed(InitSeed)
np.random.seed(InitSeed)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def scheduler(epoch, lr):
    init_lr = 0.1
    if epoch < 32:
        epoch_lr = init_lr
    elif epoch < 48:
        epoch_lr = init_lr/10
    else:
        epoch_lr = init_lr/100
    return epoch_lr

#reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
sgd = SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)

def Residual_Unit_BottleNeck(input, filter):
    if input.shape[-1] == filter[-1] or filter[0]==filter_INI:
        stride = 1
    elif input.shape[-1] != filter[-1]:
        stride = 2
    output_1 = Activation("relu")(BatchNormalization()(keras.layers.Conv2D(filter[0], (1,1), strides=stride, padding="same", use_bias=False)(input)))
    output_2 = Activation("relu")(BatchNormalization()(keras.layers.Conv2D(filter[1], (3,3), padding="same", use_bias=False)(output_1)))
    output_3 = BatchNormalization()(Conv2D(filter[2], (1,1), padding="same", use_bias=False)(output_2))
    output_res = BatchNormalization()(Conv2D(filter[2], (1, 1), strides=stride, padding="same", use_bias=False)(input))
    output = Activation("relu")(keras.layers.add([output_3,output_res]))
    return output

def Residual_Unit(input, filter):
    if input.shape[-1]==filter or filter==filter_INI:
        stride=1
    elif input.shape[-1]!=filter:
        stride=2
    output_1 = Activation("relu")(BatchNormalization()(Conv2D(filter, (3,3), strides=stride, padding="same", use_bias=False)(input)))
    output_2 = BatchNormalization()(Conv2D(filter, (3,3), padding="same", use_bias=False)(output_1))
    output_res = BatchNormalization()(Conv2D(filter, (1, 1), strides=stride, padding="same", use_bias=False)(input))
    output = Activation("relu")(add([output_2,output_res]))
    return output

#For Imagenet
#ResNet50
filter = [32, 32, 128]
times = [3, 4, 6, 3]
filter_INI = filter[0]*2
input = keras.layers.Input(shape = (224,224,3))
output_conv1 = keras.layers.Activation("relu")(keras.layers.BatchNormalization()(keras.layers.Conv2D(64, (7,7), strides=(2,2), padding="same")(input)))
output_pool1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")(output_conv1)
output_temp = output_pool1
for i in range (0, len(times)):
    filter = [number*2 for number in filter]
    for j in range (0, times[i]):
        output_temp = Residual_Unit_BottleNeck(output_temp, filter)
output_pool5 = keras.layers.GlobalAveragePooling2D()(output_temp)
output = keras.layers.Dense(10, activation="softmax")(output_pool5)
ResNet50 = keras.models.Model(inputs=input, outputs=output)
ResNet50.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#ResNet50.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, validation_batch_size=16)
#ResNet50.evaluate(x_test, y_test)

#For CIFAR10
#Transfer study (from competetion of imagenet)
baseline=keras.applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape=(32,32,3))
for layers in baseline.layers[:]:
    layers.trainable = False
output = baseline.output
output = keras.layers.GlobalAveragePooling2D()(output)
output = keras.layers.Dense(10, activation="softmax")(output)
ResNet50_Transfer = keras.models.Model(inputs=baseline.input, outputs=output)
ResNet50_Transfer.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

ResNet50_Transfer.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, validation_batch_size=16)
ResNet50_Transfer.evaluate(x_test, y_test)

#This Resnet should be simpler than the previous one, since the size of a data(32*32) is much smaller than the data size in ImageNet(224*224)
#Resnet 9n+2 n=8 (use bottleneck)
filter = [8, 8, 32]
times = [8]*3
filter_INI = filter[0]*2
input = keras.layers.Input(shape = (32,32,3))
output_conv1 = keras.layers.Activation("relu")(keras.layers.BatchNormalization()(keras.layers.Conv2D(16, (3,3), strides=1, padding="same")(input)))
output_pool1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding="same")(output_conv1)
output_temp = output_pool1
for i in range (0, len(times)):
    filter = [number*2 for number in filter]
    for j in range (0, times[i]):
        output_temp = Residual_Unit_BottleNeck(output_temp, filter)
output_pool4 = keras.layers.GlobalAveragePooling2D()(output_temp)
output = keras.layers.Dense(10, activation="softmax")(output_pool4)
ResNet50_Cifar_BN = keras.models.Model(inputs=input, outputs=output)
ResNet50_Cifar_BN.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

ResNet50_Cifar_BN.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, validation_batch_size=16)
ResNet50_Cifar_BN.evaluate(x_test, y_test)

#Resnet 6n+2 n=8 (original)
filter = 8
times = [8]*3
filter_INI = filter*2
input = keras.layers.Input(shape = (32,32,3))
output_conv1 = Activation("relu")(BatchNormalization()(Conv2D(16, (3,3), strides=1, padding="same")(input)))
output_pool1 = MaxPool2D(pool_size=(3,3), strides=1, padding="same")(output_conv1)
output_temp = output_pool1
for i in range (0, len(times)):
    filter = filter*2
    for j in range (0, times[i]):
        output_temp = Residual_Unit(output_temp, filter)
output_pool4 = GlobalAveragePooling2D()(output_temp)
output = Dense(10, activation="softmax")(output_pool4)
ResNet50_Cifar = Model(inputs=input, outputs=output)
ResNet50_Cifar.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

ResNet50_Cifar.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, validation_batch_size=16)
ResNet50_Cifar.evaluate(x_test, y_test)

'''Learning rate need to be modified'''
'''Should output a fine result'''