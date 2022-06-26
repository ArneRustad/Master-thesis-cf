import timeit
import numpy as np
import os
from tqdm.auto import tqdm
from src import constants as const

def measure_synthesizer_speed(synthesizer, synthesizer_name, nruns=1, overwrite_existing=False):
    dir_synthesizer = os.path.join(const.dir.speed_comparison(), synthesizer_name)
    os.makedirs(dir_synthesizer, exist_ok=True)
    for i in tqdm(range(nruns)):
        curr_path = os.path.join(dir_synthesizer, f"run{i}.npy")
        if overwrite_existing or not os.path.exists(curr_path):
            time = timeit.timeit(synthesizer, number=1)
            np.save(curr_path, time)

