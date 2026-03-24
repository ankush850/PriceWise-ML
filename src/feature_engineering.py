import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

def build_text_tfidf(df_train, df_test, text_col='catalog_content', n_features=5000, n_svd=128):
    vec = TfidfVectorizer(max_features=n_features, ngram_range=(1,2), stop_words='english')
    X_all = pd.concat([df_train[text_col].fillna(''), df_test[text_col].fillna('')])
    X_vec = vec.fit_transform(X_all)
    svd = TruncatedSVD(n_components=min(n_svd, X_vec.shape[1]-1), random_state=42)
    X_red = svd.fit_transform(X_vec)
    n_train = len(df_train)
    return X_red[:n_train, :], X_red[n_train:, :], vec, svd

def combine_tabular_and_emb(tab_df, text_emb, img_emb=None):
    """
    Efficiently combine tabular dataframe and embeddings (text_emb, optional img_emb)
    Avoids repeated DataFrame inserts which fragment memory; uses pd.concat instead.
    Parameters:
      - tab_df: pandas DataFrame (n_samples, k_tab)
      - text_emb: numpy array (n_samples, n_text_dims)
      - img_emb: numpy array or None (n_samples, n_img_dims)
    Returns:
      - pandas DataFrame with combined features
    """
    tab_df = tab_df.reset_index(drop=True).copy()
    parts = [tab_df]

    # text embeddings -> DataFrame with named columns
    if text_emb is not None:
        n_text = text_emb.shape[1]
        df_text = pd.DataFrame(text_emb, columns=[f"text_svd_{i}" for i in range(n_text)])
        parts.append(df_text)

    # image embeddings -> DataFrame
    if img_emb is not None:
        n_img = img_emb.shape[1]
        df_img = pd.DataFrame(img_emb, columns=[f"img_svd_{i}" for i in range(n_img)])
        parts.append(df_img)

    combined = pd.concat(parts, axis=1)
    return combined
