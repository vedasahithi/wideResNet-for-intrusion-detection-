import pandas as pd
from Main import Data_Normalization
from Main import Feature_Selection
import numpy as np
import Proposed_TWSA_WRN.Wide_Resnet
from Proposed_TWSA_WRN.Algm_Anal import Wide_Resnet_CSO
from Proposed_TWSA_WRN.Algm_Anal import Wide_Resnet_GSO
from Proposed_TWSA_WRN.Algm_Anal import Wide_Resnet_TSO
from Proposed_TWSA_WRN.Algm_Anal import Wide_Resnet_WSO
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
    Wide_Resnet_CSO.classify(attack_data, Label, kv, swarm_size, ACC, TPR, TNR)
    Wide_Resnet_GSO.classify(attack_data, Label, kv, swarm_size, ACC, TPR, TNR)
    Wide_Resnet_TSO.classify(attack_data, Label, kv, swarm_size, ACC, TPR, TNR)
    Wide_Resnet_WSO.classify(attack_data, Label, kv, swarm_size, ACC, TPR, TNR)

    return ACC,TPR,TNR
