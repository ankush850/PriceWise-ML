<<<<<<< HEAD
import pandas as pd
import numpy as np

def target_encode_smooth(train_df, test_df, col, target, smoothing=10):
    agg = train_df.groupby(col)[target].agg(['count','mean'])
    prior = train_df[target].mean()
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + smoothing * prior) / (counts + smoothing)
    mapping = smooth.to_dict()
    train_te = train_df[col].map(mapping).fillna(prior).values
    test_te = test_df[col].map(mapping).fillna(prior).values
    return train_te, test_te
=======
import pandas as pd
import numpy as np

def target_encode_smooth(train_df, test_df, col, target, smoothing=10):
    agg = train_df.groupby(col)[target].agg(['count','mean'])
    prior = train_df[target].mean()
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + smoothing * prior) / (counts + smoothing)
    mapping = smooth.to_dict()
    train_te = train_df[col].map(mapping).fillna(prior).values
    test_te = test_df[col].map(mapping).fillna(prior).values
    return train_te, test_te
>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
