<<<<<<< HEAD
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from src.data_preprocessing import basic_cleaning, add_basic_features
from src.feature_engineering import build_text_tfidf, combine_tabular_and_emb
from src.model_lightgbm import train_lgb_oof
from src.evaluate import smape

try:
    from src.advanced_features import add_advanced_features
    HAVE_ADV = True
except Exception:
    HAVE_ADV = False

# config
DATA_DIR = 'dataset'
OUT_DIR = 'outputs'
EMB_DIR = os.path.join(DATA_DIR, 'embeddings')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'submissions'), exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

def load_cached_np(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)
    return None

def build_features(df_train, df_test):
    # basic features
    df_train = add_basic_features(df_train)
    df_test = add_basic_features(df_test)

    # advanced features (if module available)
    if HAVE_ADV:
        df_train = add_advanced_features(df_train, price_col='price')
        # for test, pass price_col missing (function handles)
        df_test = add_advanced_features(df_test, price_col='price')

    # tab features to use (choose columns that exist)
    tab_cols = []
    for c in ['title_len','num_words','num_digits','ipq','ipq_missing','price_per_unit','num_top_tokens']:
        if c in df_train.columns:
            tab_cols.append(c)

    df_tab_tr = df_train[tab_cols].fillna(-1)
    df_tab_te = df_test[tab_cols].fillna(-1)

    # text embeddings: try cached SBERT first, else TF-IDF+SVD
    text_tr_emb = load_cached_np(os.path.join(EMB_DIR, 'text_train.npy'))
    text_te_emb = load_cached_np(os.path.join(EMB_DIR, 'text_test.npy'))

    if text_tr_emb is None or text_te_emb is None:
        print("Cached text embeddings not found — building TF-IDF + SVD (fast fallback).")
        X_text_tr, X_text_te, vec, svd = build_text_tfidf(df_train, df_test, n_features=5000, n_svd=128)
        joblib.dump(vec, os.path.join(OUT_DIR, 'tfidf_vec.pkl'))
        joblib.dump(svd, os.path.join(OUT_DIR, 'svd_text.pkl'))
        text_tr_emb = X_text_tr
        text_te_emb = X_text_te
    else:
        print("Loaded cached text embeddings from", EMB_DIR)

    # image embeddings: optional cached
    img_tr_emb = load_cached_np(os.path.join(EMB_DIR, 'image_train.npy'))
    img_te_emb = load_cached_np(os.path.join(EMB_DIR, 'image_test.npy'))
    if img_tr_emb is not None and img_te_emb is not None:
        print("Loaded cached image embeddings from", EMB_DIR)
    else:
        img_tr_emb = None
        img_te_emb = None

    # combine
    X_tr = combine_tabular_and_emb(df_tab_tr, text_tr_emb, img_emb=img_tr_emb)
    X_te = combine_tabular_and_emb(df_tab_te, text_te_emb, img_emb=img_te_emb)
    return X_tr, X_te, df_tab_tr, df_tab_te

def train_and_stack(X_tr, y_tr, X_te, df_test):
    oof_dict = {}
    test_preds_dict = {}

    # 1) raw price model
    print("Training LightGBM on raw price...")
    oof_raw, test_raw, models_raw, score_raw = train_lgb_oof(X_tr, y_tr, X_test=X_te, n_splits=5)
    print("OOF SMAPE (raw): {:.6f}%".format(smape(y_tr, oof_raw)))
    oof_dict['lgb_raw'] = oof_raw
    test_preds_dict['lgb_raw'] = test_raw

    # 2) log1p model
    print("Training LightGBM on log1p(price)...")
    y_log = np.log1p(y_tr)
    oof_log, test_log, models_log, _ = train_lgb_oof(X_tr, y_log, X_test=X_te, n_splits=5)
    oof_log_bt = np.expm1(oof_log)
    test_log_bt = np.expm1(test_log)
    print("OOF SMAPE (log back-trans): {:.6f}%".format(smape(y_tr, oof_log_bt)))
    oof_dict['lgb_log_bt'] = oof_log_bt
    test_preds_dict['lgb_log_bt'] = test_log_bt

    # Optionally add simple avg baseline
    oof_avg = (oof_raw + oof_log_bt) / 2.0
    test_avg = (test_raw + test_log_bt) / 2.0
    oof_dict['avg_raw_log'] = oof_avg
    test_preds_dict['avg_raw_log'] = test_avg
    print("OOF SMAPE (avg raw+log): {:.6f}%".format(smape(y_tr, oof_avg)))

    # 3) stacking meta: RidgeCV on OOFs (use original y)
    print("Training Ridge meta on OOF preds...")
    # build OOF matrix
    keys = list(oof_dict.keys())
    X_oof = np.vstack([oof_dict[k] for k in keys]).T
    X_test_meta = np.vstack([test_preds_dict[k] for k in keys]).T
    meta = RidgeCV(alphas=[0.01,0.1,1.0,10.0], cv=5).fit(X_oof, y_tr)
    test_meta_pred = meta.predict(X_test_meta)
    oof_meta_pred = meta.predict(X_oof)
    print("OOF SMAPE (Ridge meta): {:.6f}%".format(smape(y_tr, oof_meta_pred)))

    # save OOFs
    oof_df = pd.DataFrame({k: oof_dict[k] for k in oof_dict})
    oof_df['y_true'] = y_tr
    oof_df.to_csv(os.path.join(OUT_DIR, 'oof_preds.csv'), index=False)

    # final preds: clip to positive
    final_test_preds = np.maximum(0.01, test_meta_pred)
    # save submission
    out = df_test[['sample_id']].copy()
    out['price'] = final_test_preds
    out_path = os.path.join(OUT_DIR, 'submissions', 'test_out.csv')
    out.to_csv(out_path, index=False, float_format='%.6f')
    # also save alternate avg
    alt_path = os.path.join(OUT_DIR, 'submissions', 'final_submission_ensemble.csv')
    avg_out = df_test[['sample_id']].copy()
    avg_out['price'] = np.maximum(0.01, test_avg)
    avg_out.to_csv(alt_path, index=False, float_format='%.6f')

    print("Wrote submissions:", out_path, alt_path)
    return meta

def main():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Place train.csv and test.csv into dataset/ and rerun.")
        return

    print("Loading CSVs...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print("Basic cleaning...")
    df_train = basic_cleaning(df_train)
    df_test = df_test.copy()
    df_test['price'] = 0.0

    # build features
    X_tr, X_te, df_tab_tr, df_tab_te = build_features(df_train, df_test)

    # train & stack
    meta = train_and_stack(X_tr, df_train['price'].values, X_te, df_test)

    # save meta
    joblib.dump(meta, os.path.join(OUT_DIR, 'meta_ridge.pkl'))

if __name__ == '__main__':
    main()
=======
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from src.data_preprocessing import basic_cleaning, add_basic_features
from src.feature_engineering import build_text_tfidf, combine_tabular_and_emb
from src.model_lightgbm import train_lgb_oof
from src.evaluate import smape

try:
    from src.advanced_features import add_advanced_features
    HAVE_ADV = True
except Exception:
    HAVE_ADV = False

# config
DATA_DIR = 'dataset'
OUT_DIR = 'outputs'
EMB_DIR = os.path.join(DATA_DIR, 'embeddings')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'submissions'), exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

def load_cached_np(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)
    return None

def build_features(df_train, df_test):
    # basic features
    df_train = add_basic_features(df_train)
    df_test = add_basic_features(df_test)

    # advanced features (if module available)
    if HAVE_ADV:
        df_train = add_advanced_features(df_train, price_col='price')
        # for test, pass price_col missing (function handles)
        df_test = add_advanced_features(df_test, price_col='price')

    # tab features to use (choose columns that exist)
    tab_cols = []
    for c in ['title_len','num_words','num_digits','ipq','ipq_missing','price_per_unit','num_top_tokens']:
        if c in df_train.columns:
            tab_cols.append(c)

    df_tab_tr = df_train[tab_cols].fillna(-1)
    df_tab_te = df_test[tab_cols].fillna(-1)

    # text embeddings: try cached SBERT first, else TF-IDF+SVD
    text_tr_emb = load_cached_np(os.path.join(EMB_DIR, 'text_train.npy'))
    text_te_emb = load_cached_np(os.path.join(EMB_DIR, 'text_test.npy'))

    if text_tr_emb is None or text_te_emb is None:
        print("Cached text embeddings not found — building TF-IDF + SVD (fast fallback).")
        X_text_tr, X_text_te, vec, svd = build_text_tfidf(df_train, df_test, n_features=5000, n_svd=128)
        joblib.dump(vec, os.path.join(OUT_DIR, 'tfidf_vec.pkl'))
        joblib.dump(svd, os.path.join(OUT_DIR, 'svd_text.pkl'))
        text_tr_emb = X_text_tr
        text_te_emb = X_text_te
    else:
        print("Loaded cached text embeddings from", EMB_DIR)

    # image embeddings: optional cached
    img_tr_emb = load_cached_np(os.path.join(EMB_DIR, 'image_train.npy'))
    img_te_emb = load_cached_np(os.path.join(EMB_DIR, 'image_test.npy'))
    if img_tr_emb is not None and img_te_emb is not None:
        print("Loaded cached image embeddings from", EMB_DIR)
    else:
        img_tr_emb = None
        img_te_emb = None

    # combine
    X_tr = combine_tabular_and_emb(df_tab_tr, text_tr_emb, img_emb=img_tr_emb)
    X_te = combine_tabular_and_emb(df_tab_te, text_te_emb, img_emb=img_te_emb)
    return X_tr, X_te, df_tab_tr, df_tab_te

def train_and_stack(X_tr, y_tr, X_te, df_test):
    oof_dict = {}
    test_preds_dict = {}

    # 1) raw price model
    print("Training LightGBM on raw price...")
    oof_raw, test_raw, models_raw, score_raw = train_lgb_oof(X_tr, y_tr, X_test=X_te, n_splits=5)
    print("OOF SMAPE (raw): {:.6f}%".format(smape(y_tr, oof_raw)))
    oof_dict['lgb_raw'] = oof_raw
    test_preds_dict['lgb_raw'] = test_raw

    # 2) log1p model
    print("Training LightGBM on log1p(price)...")
    y_log = np.log1p(y_tr)
    oof_log, test_log, models_log, _ = train_lgb_oof(X_tr, y_log, X_test=X_te, n_splits=5)
    oof_log_bt = np.expm1(oof_log)
    test_log_bt = np.expm1(test_log)
    print("OOF SMAPE (log back-trans): {:.6f}%".format(smape(y_tr, oof_log_bt)))
    oof_dict['lgb_log_bt'] = oof_log_bt
    test_preds_dict['lgb_log_bt'] = test_log_bt

    # Optionally add simple avg baseline
    oof_avg = (oof_raw + oof_log_bt) / 2.0
    test_avg = (test_raw + test_log_bt) / 2.0
    oof_dict['avg_raw_log'] = oof_avg
    test_preds_dict['avg_raw_log'] = test_avg
    print("OOF SMAPE (avg raw+log): {:.6f}%".format(smape(y_tr, oof_avg)))

    # 3) stacking meta: RidgeCV on OOFs (use original y)
    print("Training Ridge meta on OOF preds...")
    # build OOF matrix
    keys = list(oof_dict.keys())
    X_oof = np.vstack([oof_dict[k] for k in keys]).T
    X_test_meta = np.vstack([test_preds_dict[k] for k in keys]).T
    meta = RidgeCV(alphas=[0.01,0.1,1.0,10.0], cv=5).fit(X_oof, y_tr)
    test_meta_pred = meta.predict(X_test_meta)
    oof_meta_pred = meta.predict(X_oof)
    print("OOF SMAPE (Ridge meta): {:.6f}%".format(smape(y_tr, oof_meta_pred)))

    # save OOFs
    oof_df = pd.DataFrame({k: oof_dict[k] for k in oof_dict})
    oof_df['y_true'] = y_tr
    oof_df.to_csv(os.path.join(OUT_DIR, 'oof_preds.csv'), index=False)

    # final preds: clip to positive
    final_test_preds = np.maximum(0.01, test_meta_pred)
    # save submission
    out = df_test[['sample_id']].copy()
    out['price'] = final_test_preds
    out_path = os.path.join(OUT_DIR, 'submissions', 'test_out.csv')
    out.to_csv(out_path, index=False, float_format='%.6f')
    # also save alternate avg
    alt_path = os.path.join(OUT_DIR, 'submissions', 'final_submission_ensemble.csv')
    avg_out = df_test[['sample_id']].copy()
    avg_out['price'] = np.maximum(0.01, test_avg)
    avg_out.to_csv(alt_path, index=False, float_format='%.6f')

    print("Wrote submissions:", out_path, alt_path)
    return meta

def main():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Place train.csv and test.csv into dataset/ and rerun.")
        return

    print("Loading CSVs...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print("Basic cleaning...")
    df_train = basic_cleaning(df_train)
    df_test = df_test.copy()
    df_test['price'] = 0.0

    # build features
    X_tr, X_te, df_tab_tr, df_tab_te = build_features(df_train, df_test)

    # train & stack
    meta = train_and_stack(X_tr, df_train['price'].values, X_te, df_test)

    # save meta
    joblib.dump(meta, os.path.join(OUT_DIR, 'meta_ridge.pkl'))

if __name__ == '__main__':
    main()
>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
