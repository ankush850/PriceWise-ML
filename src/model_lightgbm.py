<<<<<<< HEAD
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .evaluate import smape
import pandas as pd

def lgb_smape_eval(y_pred, train_data):
    y_true = train_data.get_label()
    y_pred = np.maximum(y_pred, 0.0)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    res = np.zeros_like(y_true, dtype=float)
    res[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 'SMAPE', float(np.mean(res)), False

def make_stratified_folds(y, n_splits=5, n_bins=10, seed=42):
    y = np.array(y)
    import pandas as pd
    y_log = np.log1p(y)
    bins = pd.qcut(y_log, q=n_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))

def train_lgb_oof(X, y, X_test=None, n_splits=5, seed=42, params=None,
                  num_boost_round=5000, early_stop_rounds=100, verbose_eval=100):
    if params is None:
        params = {"objective":"regression","metric":"None","learning_rate":0.05,"num_leaves":128,
                  "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"seed":seed}
    folds = make_stratified_folds(y, n_splits=n_splits, seed=seed)
    oof = np.zeros(len(y), dtype=float)
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    for fold, (tr_idx, val_idx) in enumerate(folds):
        print(f"Training fold {fold+1}/{len(folds)}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # use callbacks for early stopping and logging (compatible across lightgbm versions)
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop_rounds),
            lgb.log_evaluation(period=verbose_eval)
        ]

        model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                          valid_sets=[dval], feval=lgb_smape_eval,
                          callbacks=callbacks)
        best_iter = model.best_iteration if model.best_iteration is not None else num_boost_round
        oof[val_idx] = model.predict(X_val, num_iteration=best_iter)
        if X_test is not None:
            test_preds += model.predict(X_test, num_iteration=best_iter) / len(folds)
        models.append(model)

    score = smape(y, oof)
    return oof, test_preds, models, score
=======
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .evaluate import smape
import pandas as pd

def lgb_smape_eval(y_pred, train_data):
    y_true = train_data.get_label()
    y_pred = np.maximum(y_pred, 0.0)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    res = np.zeros_like(y_true, dtype=float)
    res[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 'SMAPE', float(np.mean(res)), False

def make_stratified_folds(y, n_splits=5, n_bins=10, seed=42):
    y = np.array(y)
    import pandas as pd
    y_log = np.log1p(y)
    bins = pd.qcut(y_log, q=n_bins, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))

def train_lgb_oof(X, y, X_test=None, n_splits=5, seed=42, params=None,
                  num_boost_round=5000, early_stop_rounds=100, verbose_eval=100):
    if params is None:
        params = {"objective":"regression","metric":"None","learning_rate":0.05,"num_leaves":128,
                  "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,"seed":seed}
    folds = make_stratified_folds(y, n_splits=n_splits, seed=seed)
    oof = np.zeros(len(y), dtype=float)
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    for fold, (tr_idx, val_idx) in enumerate(folds):
        print(f"Training fold {fold+1}/{len(folds)}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # use callbacks for early stopping and logging (compatible across lightgbm versions)
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop_rounds),
            lgb.log_evaluation(period=verbose_eval)
        ]

        model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                          valid_sets=[dval], feval=lgb_smape_eval,
                          callbacks=callbacks)
        best_iter = model.best_iteration if model.best_iteration is not None else num_boost_round
        oof[val_idx] = model.predict(X_val, num_iteration=best_iter)
        if X_test is not None:
            test_preds += model.predict(X_test, num_iteration=best_iter) / len(folds)
        models.append(model)

    score = smape(y, oof)
    return oof, test_preds, models, score
>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
