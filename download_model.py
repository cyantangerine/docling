from pathlib import Path
download_path = '/home/wbx/ocrdoc/models'

if __name__ == '__main__':
    from docling.utils.model_downloader import download_models
    download_models(download_path, progress=True)
    
    from huggingface_hub import snapshot_download
    print("Downloading RapidOCR models")
    from tqdm import tqdm
    snapshot_download(repo_id="SWHL/RapidOCR", local_dir=download_path, tqdm_class=tqdm)
    # print("Downloading VL models")
    # snapshot_download(repo_id='HuggingFaceTB/SmolVLM-256M-Instruct', local_dir=download_path / 'HuggingFaceTB--SmolVLM-256M-Instruct', tqdm_class=tqdm)
    # 
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct-AWQ'
    print(f"Downloading {model_name}")
    snapshot_download(repo_id=model_name, local_dir=download_path / model_name.replace('/', '--'), tqdm_class=tqdm)
    
    # model_name = 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ'
    # print(f"Downloading {model_name}")
    # snapshot_download(repo_id=model_name, local_dir=download_path / model_name.replace('/', '--'), tqdm_class=tqdm)
    