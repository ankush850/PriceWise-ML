import os, numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import resnet50
from tqdm import tqdm

def compute_and_cache_image_embs(df, img_folder, cache_path, device='cuda'):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    embs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        link = row.get('image_link','')
        fname = os.path.basename(str(link).split('?')[0])
        path = os.path.join(img_folder, fname)
        if not os.path.exists(path):
            embs.append(np.zeros(2048, dtype=np.float32))
            continue
        try:
            img = Image.open(path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).squeeze().cpu().numpy()
        except Exception:
            feat = np.zeros(2048, dtype=np.float32)
        embs.append(feat)
    embs = np.vstack(embs).astype(np.float32)
    np.save(cache_path, embs)
    return embs
