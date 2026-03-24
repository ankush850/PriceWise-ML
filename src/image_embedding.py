import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def image_to_embedding(img_path, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).squeeze().cpu().numpy()
    return feat

def compute_image_embeddings_for_df(df, img_folder, image_link_col='image_link'):
    model = load_model()
    embs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        link = row.get(image_link_col, None)
        if link is None:
            embs.append(np.zeros(2048, dtype=float))
            continue
        fname = os.path.basename(link.split('?')[0])
        local_path = os.path.join(img_folder, fname)
        if not os.path.exists(local_path):
            embs.append(np.zeros(2048, dtype=float))
            continue
        e = image_to_embedding(local_path, model)
        if e is None:
            e = np.zeros(2048, dtype=float)
        embs.append(e)
    return np.vstack(embs)