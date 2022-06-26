import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def _map_str_to_color(vec, color_map = "viridis"):
    vec_set = set(vec)
    if color_map is None:
        fig, ax = plt.subplots()
        colors = [next(ax._get_lines.prop_cycler)['color'] for i in range(len(vec_set))]
        plt.close(fig)
    else:
        cmap = matplotlib.cm.get_cmap(color_map)
        colors = cmap(np.linspace(0, 1, len(vec_set)))
    clr_map = {string: colors[i] for i, string in enumerate(sorted(vec_set))}
    return([clr_map[s] for s in vec])