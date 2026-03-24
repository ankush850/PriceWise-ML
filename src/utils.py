<<<<<<< HEAD
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(url, dest_path, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception:
        return False
    return False

def download_images(urls, dest_dir, max_workers=8):
    os.makedirs(dest_dir, exist_ok=True)
    failed = []
    def _dl(item):
        idx, url = item
        fname = os.path.basename(url.split('?')[0])
        dest = os.path.join(dest_dir, fname)
        ok = download_image(url, dest)
        return (url, ok, dest)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for url, ok, dest in exe.map(_dl, enumerate(urls)):
            if not ok:
                failed.append(url)
=======
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(url, dest_path, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception:
        return False
    return False

def download_images(urls, dest_dir, max_workers=8):
    os.makedirs(dest_dir, exist_ok=True)
    failed = []
    def _dl(item):
        idx, url = item
        fname = os.path.basename(url.split('?')[0])
        dest = os.path.join(dest_dir, fname)
        ok = download_image(url, dest)
        return (url, ok, dest)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for url, ok, dest in exe.map(_dl, enumerate(urls)):
            if not ok:
                failed.append(url)
>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
    return failed