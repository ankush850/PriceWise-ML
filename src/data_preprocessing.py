<<<<<<< HEAD
import pandas as pd
import numpy as np
import re

def load_csv_from_dataset(path):
    return pd.read_csv(path)

def basic_cleaning(df):
    # Ensure columns exist
    df = df.copy()
    if 'price' in df.columns:
        df = df[df['price'].notnull()]
        df = df[df['price'] > 0]
    # Fill missing catalog content
    if 'catalog_content' in df.columns:
        df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    return df

def extract_ipq_and_numeric_tokens(text):
    text = str(text).lower()
    # IPQ patterns: "12 pack", "12pk", "pack of 12", "12 ct", "12 count"
    m = re.search(r'(\b|^)(\d{1,4})(?:\s*(?:pack|pk|ct|count|pcs|pieces|x))', text)
    ipq = int(m.group(2)) if m else np.nan
    nums = re.findall(r'(?<!\d)(\d+\.?\d*)(?!\d)', text)
    nums = [float(x) for x in nums] if nums else []
    return ipq, nums

def add_basic_features(df):
    df = df.copy()
    df['title_len'] = df['catalog_content'].str.len()
    df['num_words'] = df['catalog_content'].str.split().apply(len)
    df['num_digits'] = df['catalog_content'].str.count(r'\d')
    # extract ipq
    ipq_list = []
    for t in df['catalog_content'].fillna(''):
        ipq, nums = extract_ipq_and_numeric_tokens(t)
        ipq_list.append(ipq)
    df['ipq'] = pd.Series(ipq_list)
    df['ipq'] = df['ipq'].fillna(-1).astype(float)
=======
import pandas as pd
import numpy as np
import re

def load_csv_from_dataset(path):
    return pd.read_csv(path)

def basic_cleaning(df):
    # Ensure columns exist
    df = df.copy()
    if 'price' in df.columns:
        df = df[df['price'].notnull()]
        df = df[df['price'] > 0]
    # Fill missing catalog content
    if 'catalog_content' in df.columns:
        df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    return df

def extract_ipq_and_numeric_tokens(text):
    text = str(text).lower()
    # IPQ patterns: "12 pack", "12pk", "pack of 12", "12 ct", "12 count"
    m = re.search(r'(\b|^)(\d{1,4})(?:\s*(?:pack|pk|ct|count|pcs|pieces|x))', text)
    ipq = int(m.group(2)) if m else np.nan
    nums = re.findall(r'(?<!\d)(\d+\.?\d*)(?!\d)', text)
    nums = [float(x) for x in nums] if nums else []
    return ipq, nums

def add_basic_features(df):
    df = df.copy()
    df['title_len'] = df['catalog_content'].str.len()
    df['num_words'] = df['catalog_content'].str.split().apply(len)
    df['num_digits'] = df['catalog_content'].str.count(r'\d')
    # extract ipq
    ipq_list = []
    for t in df['catalog_content'].fillna(''):
        ipq, nums = extract_ipq_and_numeric_tokens(t)
        ipq_list.append(ipq)
    df['ipq'] = pd.Series(ipq_list)
    df['ipq'] = df['ipq'].fillna(-1).astype(float)
>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
    return df