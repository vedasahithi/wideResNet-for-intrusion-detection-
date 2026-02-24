import keras.optimizers
from sklearn.model_selection import KFold
from keras.src.layers import Input, Add, Activation, Dropout, Flatten, Dense
#from keras.src.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.average_pooling2d import AveragePooling2D
from keras.src.layers.normalization.batch_normalization import BatchNormalization
from keras.src.regularizers import L2
from keras.src import backend as K
from Proposed_TWSA_WRN.Algm_Anal import TSO
from keras.src.layers import Input
from keras.src.models import Model
import numpy as np

weight_decay = 0.0005

def initial_conv(input):
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Conv2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    skip = Conv2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      bias_regularizer=L2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, bias_regularizer=L2(weight_decay), activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model



def classify(Data,Label,kv,swarm_size,ACC,TPR,TNR):
    Label = np.array(Label).astype('int')


    kf = KFold(n_splits=kv, shuffle=True, random_state=42)
    # Perform split
    for fold, (train_index, test_index) in enumerate(kf.split(Data), 1):
        X_train = Data[train_index]
        y_train = Label[train_index]
        X_test = Data[test_index]
        y_test = Label[test_index]
    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=1, N=2, k=2, dropout=0.0)
    wrn_28_10.compile(loss='binary_crossentropy',
                   optimizer=keras.optimizers.Adam(learning_rate=TSO.Algm(swarm_size), beta_1=0.9, beta_2=0.999),
                   metrics=['accuracy'])
    # wrn_28_10.summary()
    X_train=np.resize(X_train,(len(X_train),32,32,3))
    X_test=np.resize(X_test,(len(X_test),32,32,3))
    wrn_28_10.fit(X_train,np.array(y_train),epochs=10,verbose=0)
    pred=wrn_28_10.predict(X_test)
    prediction = np.argmax(pred, axis=1)

    target = y_test
    tp, tn, fn, fp = 1, 1, 1, 1
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(prediction)):

            if target[i] == c and prediction[i] == c:
                tp += 1
            if target[i] != c and prediction[i] != c:
                tn += 1
            if (target[i] == c and prediction[i]) != c:
                fn += 1
            if (target[i] != c and prediction[i]) == c:
                fp += 1

    Accuracy = tp + tn / (tp + tn + fp + fn)
    Tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ACC.append(Accuracy)
    TPR.append(Tpr)
    TNR.append(tnr)
    indices = [i for i, v in enumerate(prediction) if v == 0]

    return indices