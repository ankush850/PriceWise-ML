import numpy as np

def compute_text_embeddings(sentences, model_name='all-mpnet-base-v2', batch_size=64, device='cpu'):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError('sentence-transformers not available. Install with `pip install sentence-transformers`') from e
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return np.array(emb)