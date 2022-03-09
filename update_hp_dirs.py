import argparse
import os
from copy_files import transfer_files_between_folders

parser = argparse.ArgumentParser()
parser.add_argument("--filetype", help="Which file type will be extracted", type=str,
                    default=".csv")
parser.add_argument("--progress_bar", help="Bool - add progress bar or not", type=bool, default=True)
parser.add_argument("--forwards", help="Bool - whether to copy files from folder on Idun to Markov",
                    default=True)
parser.add_argument("--backwards", help="Bool - whether to copy files from folder on Markov to Idun",
                    default=False)
parser.add_argument("--verbal", "-v", help="How verbal the script should be. Enter 0 for silent.", type=int,
                    default=1)
parser.add_argument("--local", help="Backup Markov on local drive", type=bool, default=False)
parser.add_argument("--redo", help="If activated, redo copying of all files even if they already exist at destination directory",
                    type=bool, default=False)
parser.add_argument("--redo_na", help="If activated, redo copying of all datasets containing NA rows even if dataset already exist at destination directory")


idun_dir = "V:\\hyperparams_tuning"
markov_dir = "S:\\arneir\\Master-thesis-storage\\hyperparams_tuning"
local_dir = "C:\\Users\\Arne\OneDrive - NTNU\\Backup-storage-master-thesis\\hyperparams_tuning"

if __name__ == "__main__":
    args = parser.parse_args()
    kwargs = {"filetype": args.filetype, "progress_bar": args.progress_bar, "redo": args.redo, "redo_na": args.redo_na,
              "verbal": args.verbal}
    if args.forwards:
        if args.verbal >= 1:
            print("Transferring files from Idun to Markov")
        transfer_files_between_folders(idun_dir, markov_dir, **kwargs)
        if args.verbal >= 1:
            print("Finished transferring files from Idun to Markov")
    if args.backwards:
        if args.verbal >= 1:
            print("Transferring files from Markov to Idun")
        transfer_files_between_folders(markov_dir, idun_dir, **kwargs)
        if args.verbal >= 1:
            print("Finished transferring files from Markov to Idun")
    if args.local:
        if args.verbal >= 1:
            print("Transferring files from Markov to local")
        transfer_files_between_folders(markov_dir, local_dir, **kwargs)
        if args.verbal >= 1:
            print("Finished transferring files from Markov to local")
