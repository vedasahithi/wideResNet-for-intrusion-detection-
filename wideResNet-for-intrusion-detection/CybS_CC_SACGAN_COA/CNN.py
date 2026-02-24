import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import KFold  # Import train_test_split function
import numpy as np,math



def Classify(Data,Label,kv,ACC,TPR,TNR):
    Label = np.array(Label).astype('int')

    kf = KFold(n_splits=kv, shuffle=True, random_state=42)
    # Perform split
    for fold, (train_index, test_index) in enumerate(kf.split(Data), 1):
        x_train = Data[train_index]
        y_train = Label[train_index]
        x_test = Data[test_index]
        y_test = Label[test_index]


    num_classes = len(np.unique(y_train))


    xt=len(x_test)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(num_classes, activation='softmax'))

    X_test = np.resize(x_test, (xt, 28, 28, 1))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                          metrics=['accuracy'])
    pred = model.predict(X_test)

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