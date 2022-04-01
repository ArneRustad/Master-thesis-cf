import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("hp_dirs", nargs="+", help="Name of hp dirs that will be deleted")
parser.add_argument("--subfolder", help="Name of subfolder", type=str, default="")
parser.add_argument("--locations", nargs="+", help="Locations where hp directory folders will be deleted",
                    default=["Idun", "Markov"])
parser.add_argument("--verbal", "-v", help="How verbal the script should be. Enter 0 for silent.", type=int,
                    default=1)

idun_dir = "V:\\hyperparams_tuning"
markov_dir = "S:\\arneir\\Master-thesis-storage\\hyperparams_tuning"
local_dir = "C:\\Users\\Arne\OneDrive - NTNU\\Backup-storage-master-thesis\\hyperparams_tuning"
cache_dir = "S:\\arneir\\Master-thesis-storage\\python_objects"

dict_location_name_to_path = {"Markov": markov_dir, "Idun": idun_dir, "Local": local_dir}

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    for name in args.locations:
        if args.verbal >= 1:
            print(f"Deleting wanted hp_dirs on {name}")
        for hp_dir in args.hp_dirs:
            curr_hp_dir_path = os.path.join(dict_location_name_to_path[name],
                                            args.subfolder, hp_dir + "_comparison")
            if os.path.exists(curr_hp_dir_path):
                shutil.rmtree(curr_hp_dir_path)
            elif args.verbal >= 1:
                print(f"Did not find {hp_dir} subfolder at this location")
        if args.verbal >= 1:
            print(f"Finished deleting wanted hp_dirs on {name}")