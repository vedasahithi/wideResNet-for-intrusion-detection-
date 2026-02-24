from sklearn.preprocessing import MinMaxScaler
import numpy as np


def process(Data_):

    Data = Data_.loc[:, Data_.columns != "class"]
    Array = ["protocol_type", "service", "flag"]

    Label = Data_.iloc[:, -1]
    Label = np.array(Label)

    label = []
    for i in range(len(Label)):
        if Label[i] == 'normal':
            label.append(0)
        else:
            label.append(1)

    a = np.shape(Data)[1]

    Dataset = []
    Header = Data.columns
    for i in range(len(Header)):
        if (Header[i] in Array):
            str_to_convert = Data[Header[i]].values.tolist()
            union = np.unique(str_to_convert).tolist()  # Taking unique for the string present in dataset
            ind_rep = []
            for k in range(len(str_to_convert)):
                ind_rep.append(union.index(str_to_convert[k]))  # appending the index of the string
            Dataset.append(ind_rep)
        else:
            Dataset.append(Data[Header[i]].values.tolist())

    Data = np.transpose(Dataset)

    return Data,label



def min_max_norm(data):
    data,label=process(data)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_data = np.array(normalized_data)

    return normalized_data,label
