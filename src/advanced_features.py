import re, numpy as np, pandas as pd
from collections import Counter

def parse_ipq(text):
    text = str(text).lower()
    # catch patterns like "12 pack", "12pk", "pack of 12", "12 ct", "12 count", "x12", "12 pcs"
    m = re.search(r'(?:pack of|pack|pk|ct|count|pcs|pieces|x)\D*(\d{1,4})', text)
    if not m:
        # fallback: digits followed by 'pk' or 'pack' etc attached
        m = re.search(r'(\d{1,4})\s*(?:pk|pack|ct|count|pcs|pieces)\b', text)
    if m:
        try:
            return int(m.group(1))
        except:
            return np.nan
    m2 = re.search(r'\b(\d{1,4})\s*[xX]\b', text)
    if m2:
        return int(m2.group(1))
    return np.nan

def add_advanced_features(df, price_col='price'):
    df = df.copy()
    df['ipq'] = df['catalog_content'].fillna('').map(parse_ipq).fillna(-1).astype(float)
    df['ipq_missing'] = (df['ipq'] <= 0).astype(int)
    if price_col in df.columns:
        df['price_per_unit'] = df[price_col] / df['ipq'].replace({0:np.nan, -1:np.nan})
        df['price_per_unit'] = df['price_per_unit'].fillna(df[price_col].median())
    else:
        df['price_per_unit'] = np.nan
    tokens = df['catalog_content'].fillna('').str.lower().str.split()
    all_tokens = Counter([t for seq in tokens for t in seq if len(t)>2])
    topk = set([w for w, _ in all_tokens.most_common(200)])
    def top_token_feats(seq):
        seq = seq or []
        s = set(seq)
        return [1 if tok in s else 0 for tok in topk]
    # cheaper: create counts of top-words presence, or create document frequency features
    df['num_top_tokens'] = tokens.map(lambda seq: sum(1 for t in seq if t in topk))
    return df
