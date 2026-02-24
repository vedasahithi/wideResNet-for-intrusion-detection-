import pandas as pd
import Data_Normalization
import Feature_Selection
import numpy as np
import Proposed_TWSA_WRN.Wide_Resnet
import CybS_CC_SACGAN_COA.CNN
import FEFS_DLM.DNN
import HSBOA_QRNN.QRNN
import IMFL_IDSCS.NN
import RF_RBFNN.RNN

def callmain(fea_dim,kv):
    ACC1,TPR1,TNR1=[],[],[]
    ACC,TPR,TNR=[],[],[]

    data=pd.read_csv("Dataset/NSL_KDD.csv", nrows=1000)
    ######## Data Normalization using Min Max #########
    Norm_data,Label=Data_Normalization.min_max_norm(data)

    n_fs=10  # Number of features to be selected
    swarm_size=20
    ###### feature selection using SVM-RFE with HSBOA ####
    sel_fea=Feature_Selection.callmain(Norm_data,Label,n_fs)
    sel_fea=sel_fea[:,0:fea_dim]
    ########################## Proposed Attack detection and Mitigation
    attack_indx=Proposed_TWSA_WRN.Wide_Resnet.classify(sel_fea, Label,kv,swarm_size,ACC1,TPR1,TNR1)
    attack_data=[sel_fea[indx] for indx in attack_indx]
    attack_data=np.array(attack_data)
    Proposed_TWSA_WRN.Wide_Resnet.classify(attack_data, Label, kv, swarm_size, ACC, TPR, TNR)
    ##########################  Compartaive Methods #############
    CybS_CC_SACGAN_COA.CNN.Classify(attack_data, Label, kv, ACC, TPR, TNR)
    FEFS_DLM.DNN.Classify(attack_data, Label, kv, ACC, TPR, TNR)
    HSBOA_QRNN.QRNN.Classify(attack_data, Label, kv, ACC, TPR, TNR)
    IMFL_IDSCS.NN.Classify(attack_data, Label, kv, ACC, TPR, TNR)
    RF_RBFNN.RNN.classify(attack_data, Label, kv, ACC, TPR, TNR)

    return ACC,TPR,TNR
