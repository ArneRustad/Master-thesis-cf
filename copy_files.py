import os
from tqdm.auto import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", help="Source directory", type=str)
parser.add_argument("dst_dir", help="Destination directory", type=str)
parser.add_argument("--filetype", help="Which file type will be extracted", type=str)
parser.add_argument("--progress_bar", help="Bool - add progress bar or not", type=bool,
                    default="", required=False)

def extract_relative_filepaths(directory, filetype = ""):
    file_relpaths_of_interest = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        relative_dirpath = os.path.relpath(dirpath, directory)
        file_relpaths_of_interest.update([os.path.join(relative_dirpath, f) for f in filenames if f.endswith(filetype)])
    return file_relpaths_of_interest

def transfer_files_between_folders(src_dir, dst_dir, filetype="", progress_bar=True):
    file_relpaths_of_interest_src = extract_relative_filepaths(src_dir, filetype=filetype)
    file_relpaths_of_interest_dst = extract_relative_filepaths(dst_dir, filetype=filetype)
    file_relpaths_of_interest = file_relpaths_of_interest_src.difference(file_relpaths_of_interest_dst)

    with tqdm(total=len(file_relpaths_of_interest), disable=not progress_bar) as pbar:
        for file_relpath in file_relpaths_of_interest:
            filepath_dst = os.path.join(src_dir, file_relpath)
            filepath_src = os.path.join(dst_dir, file_relpath)
            os.makedirs(os.path.dirname(filepath_dst), exist_ok=True)
            shutil.copyfile(filepath_src, filepath_dst)
            pbar.update(1)

if __name__ == "__main__":
    args = parser.parse_args()
    transfer_files_between_folders(args.src_dir, args.dst_dir, filetype=args.filetype, progress_bar=args.progress_bar)