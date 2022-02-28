import argparse
import os
from copy_files import transfer_files_between_folders

parser = argparse.ArgumentParser()
parser.add_argument("--filetype", help="Which file type will be extracted", type=str,
                    default=".csv")
parser.add_argument("--progress_bar", help="Bool - add progress bar or not", type=bool)
parser.add_argument("--forwards", help="Bool - whether to copy files from folder on Idun to Markov",
                    default=True)
parser.add_argument("--backwards", help="Bool - whether to copy files from folder on Markov to Idun",
                    default=False)
parser.add_argument("--verbal", help="How verbal the script should be. Enter 0 for silent.", type=int,
                    default=1)
args = parser.parse_args()

src_dir = "V:\\hyperparams_tuning"
dst_dir = "S:\\arneir\\Master-thesis-storage\\hyperparams_tuning"

if args.forwards:
    if args.verbal >= 1:
        print("Transferring files from Idun to Markov")
    transfer_files_between_folders(src_dir, dst_dir, filetype=args.filetype, progress_bar=args.progress_bar)
    if args.verbal >= 1:
        print("Finished transferring files from Idun to Markov")
if args.backwards:
    if args.verbal >= 1:
        print("Transferring files from Markov to Idun")
    transfer_files_between_folders(dst_dir, src_dir, filetype=args.filetype, progress_bar=args.progress_bar)
    if args.verbal >= 1:
        print("Finished transferring files from Markov to Idun")