import os
from tqdm.auto import tqdm
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("src_dir", help="Source directory", type=str)
parser.add_argument("dst_dir", help="Destination directory", type=str)
parser.add_argument("--file_type", help="Which file type will be extracted", type=str)
parser.add_argument("--progress_bar", help="Bool - add progress bar or not", type=bool,
                    default="", required=False)
args = parser.parse_args()

orig_dir = "V:\\hyperparams_tuning"
new_dir = "S:\\arneir\\Master-thesis-storage\\hyperparams_tuning"
filetype = ".csv"
progress_bar=True