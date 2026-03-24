import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

def train_meta_and_predict(oof_preds_dict, test_preds_dict, y_true, alphas=[0.1,1.0,10.0]):
    """
    oof_preds_dict: {'modelA': oof_array, 'modelB': oof_array, ...}
    test_preds_dict: {'modelA': test_array, ...}
    """
    X_oof = np.vstack([oof_preds_dict[k] for k in oof_preds_dict.keys()]).T
    X_test = np.vstack([test_preds_dict[k] for k in test_preds_dict.keys()]).T
    meta = RidgeCV(alphas=alphas, cv=5)
    meta.fit(X_oof, y_true)
    test_meta = meta.predict(X_test)
    return meta, test_meta
