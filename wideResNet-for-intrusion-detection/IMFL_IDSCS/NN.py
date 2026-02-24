import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def Classify(Data,Label,kv,ACC,TPR,TNR):
    Label = np.array(Label).astype('int')

    kf = KFold(n_splits=kv, shuffle=True, random_state=42)
    # Perform split
    for fold, (train_index, test_index) in enumerate(kf.split(Data), 1):
        x_train = Data[train_index]
        y_train = Label[train_index]
        x_test = Data[test_index]
        y_test = Label[test_index]


    nc=len(np.unique(y_train))
    # Defining Model
    # Using Sequential() to build layers one after another
    model = tf.keras.Sequential([

        # Flatten Layer that converts images to 1D array
        tf.keras.layers.Flatten(),

        # Hidden Layer with 512 units and relu activation
        tf.keras.layers.Dense(units=512, activation='relu'),

        # Output Layer with 10 units for 10 classes and softmax activation
        tf.keras.layers.Dense(units=nc, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    x_test=np.array(x_test)	

    # Making Predictions
    pred = model.predict(x_test)

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







