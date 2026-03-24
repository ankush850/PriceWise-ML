import os, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def compute_and_cache_text_embs(texts, cache_path='dataset/embeddings/text_emb.npy',
                                model_name='all-mpnet-base-v2', batch_size=64, device='cuda'):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    np.save(cache_path, emb)
    return emb
