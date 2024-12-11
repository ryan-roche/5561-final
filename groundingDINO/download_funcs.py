# Helper functions for downloading files

import requests
from pathlib import Path
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd


def download_single_file(url_filename, output_dir, chunk_size=8192):
    url, filename = url_filename
    
    # Skip empty or invalid URLs
    if pd.isna(url) or not isinstance(url, str):
        return False, f"Invalid URL: {url}"
    
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with tqdm(total=total_size, unit='B', unit_scale=True, position=1, leave=False) as pbar:
        pbar.set_description(f"Downloading {filename}")
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return True, output_path

def batch_download_files(urls, output_dir="downloads", chunk_size=8192):
  """
  Downloads a list of files with tqdm progress bars
  
  params:
    urls: list of 2-tuples consisting of the url and output filename of each file
    output_dir: absolute path to the directory the output files should be written to
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Filter out any empty or NaN values
  filtered_urls = []
  for url_filename in urls:
      url, filename = url_filename
      if pd.notna(url) and isinstance(url, str):
          filtered_urls.append((url, filename))
  
  if not filtered_urls:
      print("No valid URLs found to download")
      return [], []
  
  successful_downloads = []
  failed_downloads = []
  
  with tqdm(total=len(filtered_urls), desc="Overall Progress", position=0) as pbar:
      for url_filename in filtered_urls:
          try:
              success, result = download_single_file(url_filename, output_dir, chunk_size)
              if success:
                  successful_downloads.append(result)
              else:
                  failed_downloads.append(result)
          except Exception as e:
              failed_downloads.append(str(e))
          pbar.update(1)
  
  print(f"\nCompleted: {len(successful_downloads)} files")
  if failed_downloads:
      print(f"Failed: {len(failed_downloads)} files")
      for error in failed_downloads:
          print(error)
  
  return successful_downloads, failed_downloads
