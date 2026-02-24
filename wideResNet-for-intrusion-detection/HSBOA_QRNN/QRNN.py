from sklearn.model_selection import KFold
from HSBOA_QRNN.tf_qrnn import QRNN
import numpy as np


def Classify(Data,Label,kv,ACC,TPR,TNR):
    Label = np.array(Label).astype('int')

    kf = KFold(n_splits=kv, shuffle=True, random_state=42)
    # Perform split
    for fold, (train_index, test_index) in enumerate(kf.split(Data), 1):
        x_train = Data[train_index]
        y_train = Label[train_index]
        x_test = Data[test_index]
        y_test = Label[test_index]
    word_size = 10
    size = 5
    #data = self.create_test_data(batch_size, sentence_length, word_size)


    qrnn = QRNN(in_size=word_size, size=size, conv_size=1)

    prediction = qrnn.forward(x_test)

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





