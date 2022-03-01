import os
from tqdm.auto import tqdm
import shutil
import argparse
# import pandas #Only used if --redo_na is called

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", help="Source directory", type=str)
parser.add_argument("dst_dir", help="Destination directory", type=str)
parser.add_argument("--filetype", help="Which file type will be extracted", type=str)
parser.add_argument("--progress_bar", help="Bool - add progress bar or not", type=bool,
                    default="")
parser.add_argument("--redo", help="If activated, redo copying of all files even if they already exist at destination directory",
                    type=bool, default=False)
parser.add_argument("--redo_na", help="If activated, redo copying of all datasets containing NA rows even if dataset already exist at destination directory")

def extract_relative_filepaths(directory, filetype = ""):
    file_relpaths_of_interest = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        relative_dirpath = os.path.relpath(dirpath, directory)
        file_relpaths_of_interest.update([os.path.join(relative_dirpath, f) for f in filenames if f.endswith(filetype)])
    return file_relpaths_of_interest

def check_for_na_row(path, dir="", filetypes_allowed=[".csv"]):
    import pandas as pd
    file = os.path.join(dir, path)
    df_file_type = False
    for fileype in filetypes_allowed:
        if file.endswith(fileype):
            df_file_type = True
            break
    if df_file_type:
        data = pd.read_csv(file)
        if data.isna().sum().sum() >= 1:
            return True
    # if not file found to be a dataset and contain NA observations, then return False
    return False

def transfer_files_between_folders(src_dir, dst_dir, filetype="", progress_bar=True, redo=False, redo_na=False,
                                   verbal=1):
    file_relpaths_of_interest_src = extract_relative_filepaths(src_dir, filetype=filetype)
    if redo:
        file_relpaths_of_interest = file_relpaths_of_interest_src
    else:
        file_relpaths_of_interest_dst = extract_relative_filepaths(dst_dir, filetype=filetype)
        file_relpaths_of_interest = file_relpaths_of_interest_src.difference(file_relpaths_of_interest_dst)

    if redo_na and not redo:
        if verbal >= 1:
            print("Checking for datasets with NA observations in destination directory")
        file_relpaths_with_na = [f for f in file_relpaths_of_interest_dst if check_for_na_row(f, dir=dst_dir)]
        n_na_datasets = len(file_relpaths_with_na)
        if verbal >= 1:
            print(f"Found {n_na_datasets} datset(s) with NA observations at destination directory")
        file_relpaths_of_interest.update(file_relpaths_with_na)

    if verbal >= 1:
        print(f"Found {len(file_relpaths_of_interest)} files to copy")
    with tqdm(total=len(file_relpaths_of_interest), disable=not progress_bar) as pbar:
        for file_relpath in file_relpaths_of_interest:
            filepath_src = os.path.join(src_dir, file_relpath)
            filepath_dst = os.path.join(dst_dir, file_relpath)
            os.makedirs(os.path.dirname(filepath_dst), exist_ok=True)
            shutil.copyfile(filepath_src, filepath_dst)
            pbar.update(1)

if __name__ == "__main__":
    args = parser.parse_args()
    transfer_files_between_folders(args.src_dir, args.dst_dir, filetype=args.filetype, progress_bar=args.progress_bar)