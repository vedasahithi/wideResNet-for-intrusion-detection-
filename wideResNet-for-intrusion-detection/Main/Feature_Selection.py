from sklearn.feature_selection import RFE
import HSBOA
from sklearn.svm import SVR
import numpy as np

def callmain(Features,Label,fs):

    estimator = SVR(kernel="linear",epsilon=HSBOA.Algm())
    selector = RFE(estimator, n_features_to_select=fs, step=1)
    selector = selector.fit(Features, Label)

    sel_rank=selector.ranking_
    sel_rank=sel_rank[0:fs]

    Sel_Features=Features[:,sel_rank]

    return Sel_Features
