import plotnine as p9
import os
import numpy as np
import pandas as pd
from src import constants as const
def eval_synthesizer_speed(synthesizer_names, nruns=1, plot=False, save_dir=None, save_path=None,
                           rotation_x_axis_labels=0, hjust_x_axis_labels=0.5,
                           figsize=(12,8)):
    average_training_times = np.empty(len(synthesizer_names), dtype=float)
    for i, synthesizer_name in enumerate(synthesizer_names):
        curr_training_times = []
        for j in range(nruns):
            curr_path = os.path.join(const.dir.speed_comparison(), synthesizer_name, f"run{j}.npy")
            if os.path.exists(curr_path):
                curr_training_times.append(np.load(curr_path))
            else:
                print(f"Synthesizer {synthesizer_name} only has {j} run(s).")
                break
        average_training_times[i] = np.mean(curr_training_times)
    if plot:
        p = p9.ggplot() + p9.geom_col(p9.aes(x=synthesizer_names, y=average_training_times))
        p += p9.geom_text(p9.aes(x=synthesizer_names, y=average_training_times,
                                 label=p9.after_stat(np.round(average_training_times).astype(int))),
                          nudge_y=np.max(average_training_times)*0.03)
        p += p9.xlab("Method") + p9.ylab("Time [seconds]")
        p += p9.theme(figure_size=figsize,
                      axis_text_x=p9.element_text(rotation=rotation_x_axis_labels,
                                                  hjust=hjust_x_axis_labels))
        print(p)

        if save_path is not None:
            if save_dir is not None:
                save_path = os.path.join(save_dir, save_path)
            p.save(filename=save_path, width=figsize[0], height=figsize[1], units="in")
    return pd.DataFrame({"Method": synthesizer_names, "Time": average_training_times})