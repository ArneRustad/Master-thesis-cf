import os
from tqdm.auto import tqdm
import shutil
import argparse
import pickle
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
parser.add_argument("--cache_dir", help="Directory where a pickle file with current activate path will be stored",
                    type=str, default="")
parser.add_argument("--update_if_newer", help="Update file in destination directory if file in source directory is newer",
                    type=bool, default=False)
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
                                   cache_dir="", cache_name="file_transfer_active_path.pkl", verbal=1,
                                   add_rel_path="", update_if_newer=False):
    if add_rel_path:
        src_dir = os.path.join(src_dir, add_rel_path)
        dst_dir = os.path.join(dst_dir, add_rel_path)
    
    os.makedirs(dst_dir, exist_ok=True)

    # Check if unresolved copy from last transfer
    assert cache_name[-4:] == ".pkl"
    if verbal >= 1:
        print("Checking for unresolved copy from last transfer")
    if cache_dir:
        cache_active_path_file = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_active_path_file):
            with open(cache_active_path_file, 'rb') as file:
                potential_activate_path = pickle.load(file)
            if potential_activate_path and os.path.exists(potential_activate_path):
                os.remove(potential_activate_path)
                if verbal >= 1:
                    print("Found and deleted unresolved copy from last transfer")

    if verbal >= 1:
        print("Determining file paths for files to copy")
    file_relpaths_of_interest_src = extract_relative_filepaths(src_dir, filetype=filetype)
    if redo:
        file_relpaths_of_interest = file_relpaths_of_interest_src
    else:
        file_relpaths_of_interest_dst = extract_relative_filepaths(dst_dir, filetype=filetype)
        file_relpaths_of_interest = file_relpaths_of_interest_src.difference(file_relpaths_of_interest_dst)
        if update_if_newer:
            file_relpaths_of_interest_present_in_both = file_relpaths_of_interest_src.intersection(
                file_relpaths_of_interest_dst
            )
            mtime_src_files = [os.path.getmtime(os.path.join(src_dir, rel_path))
                               for rel_path in file_relpaths_of_interest_present_in_both]
            mtime_dst_files = [os.path.getmtime(os.path.join(dst_dir, rel_path))
                                                for rel_path in file_relpaths_of_interest_present_in_both]
            files_to_update_relpaths = {
                rel_path if mtime_src > mtime_dst else None for rel_path, mtime_src, mtime_dst in
                zip(file_relpaths_of_interest_present_in_both, mtime_src_files, mtime_dst_files)
            }
            files_to_update_relpaths.remove(None)
            file_relpaths_of_interest = file_relpaths_of_interest.union(files_to_update_relpaths)

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
            if cache_dir:
                with open(cache_active_path_file, 'wb') as file:
                    pickle.dump(filepath_dst, file)
            shutil.copyfile(filepath_src, filepath_dst)
            if cache_dir:
                with open(cache_active_path_file, 'wb') as file:
                    pickle.dump("", file)
            pbar.update(1)

if __name__ == "__main__":
    args = parser.parse_args()
    transfer_files_between_folders(args.src_dir, args.dst_dir, filetype=args.filetype, progress_bar=args.progress_bar,
                                   cache_dir=args.cache_dir, update_if_newer=args.update_if_newer)