# Dataloader for TACO images

import os
import json
import pandas as pd
import torch
from download_funcs import download_single_file, batch_download_files
from groundingdino.util.inference import load_image
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from typing import Tuple, Dict


class TACODataset(Dataset):
  def __init__(self, json_path, imgs_path, sample_count=None):
    self.imgs_path = imgs_path
    
    # Load JSON data from file
    with open(json_path, 'r') as fp:
      json_data = json.load(fp)

    # Get a list of category/supercategory names
    self.categories = []
    self.supercategories = []
    for cat in tqdm(json_data['categories'], desc="Processing Categories", leave=True):
      self.categories.append(cat['name'])
      supercat = cat['supercategory']
      if supercat not in self.supercategories:
        self.supercategories.append(supercat)

    # Get the first 'sample_count' annotations.
    # If no value was set for 'sample_count', get all of them.
    self.annotations = json_data['annotations'][:sample_count] if sample_count is not None else json_data['annotations']

    # Get list of image filenames
    self._image_filenames = [img['file_name'] for img in json_data['images']]

    self._bboxes = {}   # Dictionary of {filename: bboxes} pairs
    self._labels = {}   # Dictionary of {filename: [str]} pairs
    for a in tqdm(self.annotations, desc="Processing Annotations", leave=True):
      try:
        filename = self._image_filenames[a['image_id']]
      except Exception as e:
        print(a['id'])
        raise e
      bbox = a['bbox']
      if filename in self._bboxes:
        self._bboxes[filename] = torch.vstack((self._bboxes[filename], torch.tensor(bbox)))
      else:
        self._bboxes[filename] = torch.tensor(bbox)
      if filename in self._labels:
        self._labels[filename].append(json_data['categories'][a['category_id']]['supercategory'])
      else:
        self._labels[filename] = [json_data['categories'][a['category_id']]['supercategory']]

    return


  def __len__(self):
    return len(self._bboxes)


  def __getitem__(self, idx):
    _, image = load_image(os.path.join(self.imgs_path, self._image_filenames[idx]))
    bboxes = self._bboxes[self._image_filenames[idx]]
    labels = self._labels[self._image_filenames[idx]]

    return image, bboxes, labels



class TACODownloader:

  def __init__(self, repo_path, download_dir, use_full_resolution=False):
    """
    repo_path: path to the downloaded TACO repository
    download_dir: location to download images to
    """
    self.repo_path = repo_path
    self.download_dir = download_dir
    self.use_full_resolution = use_full_resolution

  
  def check_params(self):
    """
    Verifies that valid path parameters were given
    """
    def _verify_directory_and_files(directory):
      # Check if the directory exists
      if os.path.exists(directory) and os.path.isdir(directory):
          # Check if the directory contains files
          if os.listdir(directory):  # This returns a list of files and directories in the directory
              return True  # Directory exists and contains files
          else:
              return False  # Directory is empty
      else:
          return False  # Directory doesn't exist

    if _verify_directory_and_files(self.repo_path):
      if os.path.exists(self.download_dir) and os.path.isdir(self.download_dir):
        return True
    
    return False


  def download_images(self):

    def _load_csv():
      DICT_FILE = 'annotations.json'
      DICT_PATH = os.path.join(self.repo_path, 'data', DICT_FILE)
      if not os.path.exists(DICT_PATH):
        raise FileNotFoundError(f"{DICT_FILE} not found!")
      print(f"✅ Found {DICT_FILE}!")

      with open(DICT_PATH, 'r') as fp:
        data = json.load(fp)

      return data


    def _download_files(json_data):
      print('🗄️ Processing image URLs', flush=True)
      image_df = pd.DataFrame.from_dict(json_data['images'])
      
      url_column = 'flickr_url' if self.use_full_resolution else 'flickr_640_url'
      urls_and_names = image_df[[url_column, 'file_name']].to_dict('records')
      
      print("Debug - first record:", urls_and_names[0])
      
      print(f'🔎 Found {len(urls_and_names)} Files')
      
      print("📥 Beginning downloads")
      urls_list = []
      for item in urls_and_names:
          url = item[url_column]
          filename = item['file_name']
          urls_list.append((url, filename))
          
      batch_download_files(urls=urls_list, output_dir=self.download_dir)


    json_data = _load_csv()
    print()
    _download_files(json_data)


  @staticmethod
  def test_import():
    """
    Prints a message - used to verify class import
    """
    print("DataLoader imported successfully!")

