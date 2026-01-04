import argparse
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置区域 =================
BASE_URL = "https://data.commoncrawl.org"
SAVE_ROOT = "/dev/shm/data/Nemotron-CC"  # 本地保存的根目录
MAX_WORKERS = 64            # 并发下载数
# ===========================================

def parse_and_download(path):
    path = path.strip()
    if not path: return

    # 1. 使用正则表达式提取关键信息 (quality, kind, kind2 和文件名)
    # 模式匹配：找 quality=xxx, kind=xxx, kind2=xxx 以及最后的文件名
    pattern = r"quality=([^/]+)/kind=([^/]+)/kind2=([^/]+)/([^/]+)$"
    match = re.search(pattern, path)

    if not match:
        return f"Skipped (Format mismatch): {path}"

    quality_val, kind_val, kind2_val, filename = match.groups()

    # 2. 构建本地三级目录结构
    # 方案 A: 保留 key=value (推荐，数据分析工具友好) -> downloads/quality=high/kind=actual/...
    relative_dir = os.path.join(f"quality={quality_val}", f"kind={kind_val}", f"kind2={kind2_val}")
    
    # 方案 B: 如果你只想要值 (downloads/high/actual/actual/...)，请注释上面，解开下面这行：
    # relative_dir = os.path.join(quality_val, kind_val, kind2_val)

    local_dir = os.path.join(SAVE_ROOT, relative_dir)
    local_file_path = os.path.join(local_dir, filename)

    # 3. 创建目录
    os.makedirs(local_dir, exist_ok=True)

    # 4. 检查文件是否已存在 (断点续传简单判断)
    if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
        return None  # 已存在，跳过

    # 5. 下载文件
    download_url = f"{BASE_URL}/{path}"
    try:
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            # 使用临时文件写入，避免中断导致文件损坏
            temp_path = local_file_path + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # 下载成功后重命名
            os.rename(temp_path, local_file_path)
            return None
    except Exception as e:
        return f"Error downloading {filename}: {str(e)}"

def main(shard=None, shard_number=None):
    # 如果你的路径在文件中，取消下面注释
    with open("paths.txt", "r") as f:
        url_paths = f.readlines()

    url_paths = [url for url in url_paths if "/quality=high/" in url]

    if shard is not None and shard_number is not None:
        total_files = len(url_paths)
        shard_size = (total_files + shard_number - 1) // shard_number
        start_index = shard * shard_size
        end_index = min(start_index + shard_size, total_files)
        url_paths = url_paths[start_index:end_index]

    print(f"开始处理 {len(url_paths)} 个文件...")
    print(f"保存路径: {os.path.abspath(SAVE_ROOT)}")

    # 进度条配置
    pbar = tqdm(total=len(url_paths), unit="file")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_and_download, url): url for url in url_paths}
        
        for future in futures:
            result = future.result()
            if result:
                pbar.write(result) # 打印错误日志，不破坏进度条
            pbar.update(1)
            
    pbar.close()
    print("全部完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=None, help="当前分片编号 (0-based)")
    parser.add_argument("--shard_number", type=int, default=None, help="总分片数")
    args = parser.parse_args()
    main(shard=args.shard, shard_number=args.shard_number)
